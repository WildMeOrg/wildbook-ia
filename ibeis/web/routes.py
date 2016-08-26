# -*- coding: utf-8 -*-
"""
Dependencies: flask, tornado
"""
from __future__ import absolute_import, division, print_function
import random
import math
from flask import request, current_app, url_for
from ibeis.control import controller_inject
from ibeis import constants as const
from ibeis.constants import KEY_DEFAULTS, SPECIES_KEY
from ibeis.web import appfuncs as appf
from ibeis.web import routes_ajax
import utool as ut
import vtool as vt
import numpy as np

register_route = controller_inject.get_ibeis_flask_route(__name__)


@register_route('/', methods=['GET'])
def root():
    return appf.template(None)


@register_route('/view/', methods=['GET'])
def view():
    def _date_list(gid_list):
        unixtime_list = ibs.get_image_unixtime(gid_list)
        datetime_list = [
            ut.unixtime_to_datetimestr(unixtime)
            if unixtime is not None else
            'UNKNOWN'
            for unixtime in unixtime_list
        ]
        datetime_split_list = [ datetime.split(' ') for datetime in datetime_list ]
        date_list = [ datetime_split[0] if len(datetime_split) == 2 else 'UNKNOWN' for datetime_split in datetime_split_list ]
        return date_list

    def filter_annots_imageset(aid_list):
        try:
            imgsetid = request.args.get('imgsetid', '')
            imgsetid = int(imgsetid)
            imgsetid_list = ibs.get_valid_imgsetids()
            assert imgsetid in imgsetid_list
        except:
            print('ERROR PARSING IMAGESET ID FOR ANNOTATION FILTERING')
            return aid_list
        imgsetids_list = ibs.get_annot_imgsetids(aid_list)
        aid_list = [
            aid
            for aid, imgsetid_list_ in zip(aid_list, imgsetids_list)
            if imgsetid in imgsetid_list_
        ]
        return aid_list

    def filter_images_imageset(gid_list):
        try:
            imgsetid = request.args.get('imgsetid', '')
            imgsetid = int(imgsetid)
            imgsetid_list = ibs.get_valid_imgsetids()
            assert imgsetid in imgsetid_list
        except:
            print('ERROR PARSING IMAGESET ID FOR IMAGE FILTERING')
            return gid_list
        imgsetids_list = ibs.get_image_imgsetids(gid_list)
        gid_list = [
            gid
            for gid, imgsetid_list_ in zip(gid_list, imgsetids_list)
            if imgsetid in imgsetid_list_
        ]
        return gid_list

    def filter_names_imageset(nid_list):
        try:
            imgsetid = request.args.get('imgsetid', '')
            imgsetid = int(imgsetid)
            imgsetid_list = ibs.get_valid_imgsetids()
            assert imgsetid in imgsetid_list
        except:
            print('ERROR PARSING IMAGESET ID FOR ANNOTATION FILTERING')
            return nid_list
        aids_list = ibs.get_name_aids(nid_list)
        imgsetids_list = [
            set(ut.flatten(ibs.get_annot_imgsetids(aid_list)))
            for aid_list in aids_list
        ]
        nid_list = [
            nid
            for nid, imgsetid_list_ in zip(nid_list, imgsetids_list)
            if imgsetid in imgsetid_list_
        ]
        return nid_list

    ibs = current_app.ibs

    filter_kw = {
        'multiple': None,
        'minqual': 'good',
        'is_known': True,
        'min_pername': 1,
        'view': ['right'],
    }

    aid_list = ibs.get_valid_aids()
    aid_list = ibs.filter_annots_general(aid_list, filter_kw=filter_kw)
    aid_list = filter_annots_imageset(aid_list)
    gid_list = ibs.get_annot_gids(aid_list)
    unixtime_list = ibs.get_image_unixtime(gid_list)
    nid_list = ibs.get_annot_name_rowids(aid_list)
    date_list = _date_list(gid_list)

    flagged_date_list = ['2016/01/29', '2016/01/30', '2016/01/31', '2016/02/01']

    gid_list_unique = list(set(gid_list))
    date_list_unique = _date_list(gid_list_unique)
    date_taken_dict = {}
    for gid, date in zip(gid_list_unique, date_list_unique):
        if date not in flagged_date_list:
            continue
        if date not in date_taken_dict:
            date_taken_dict[date] = [0, 0]
        date_taken_dict[date][1] += 1

    gid_list_all = ibs.get_valid_gids()
    gid_list_all = filter_images_imageset(gid_list_all)
    date_list_all = _date_list(gid_list_all)
    for gid, date in zip(gid_list_all, date_list_all):
        if date not in flagged_date_list:
            continue
        if date in date_taken_dict:
            date_taken_dict[date][0] += 1

    value = 0
    label_list = []
    value_list = []
    index_list = []
    seen_set = set()
    current_seen_set = set()
    previous_seen_set = set()
    last_date = None
    date_seen_dict = {}
    for index, (unixtime, aid, nid, date) in enumerate(sorted(zip(unixtime_list, aid_list, nid_list, date_list))):
        if date not in flagged_date_list:
            continue

        index_list.append(index + 1)
        # Add to counters

        if date not in date_seen_dict:
            date_seen_dict[date] = [0, 0, 0, 0]

        date_seen_dict[date][0] += 1

        if nid not in current_seen_set:
            current_seen_set.add(nid)
            date_seen_dict[date][1] += 1
            if nid in previous_seen_set:
                date_seen_dict[date][3] += 1

        if nid not in seen_set:
            seen_set.add(nid)
            value += 1
            date_seen_dict[date][2] += 1

        # Add to register
        value_list.append(value)
        # Reset step (per day)
        if date != last_date and date != 'UNKNOWN':
            last_date = date
            previous_seen_set = set(current_seen_set)
            current_seen_set = set()
            label_list.append(date)
        else:
            label_list.append('')

    # def optimization1(x, a, b, c):
    #     return a * np.log(b * x) + c

    # def optimization2(x, a, b, c):
    #     return a * np.sqrt(x) ** b + c

    # def optimization3(x, a, b, c):
    #     return 1.0 / (a * np.exp(-b * x) + c)

    # def process(func, opts, domain, zero_index, zero_value):
    #     values = func(domain, *opts)
    #     diff = values[zero_index] - zero_value
    #     values -= diff
    #     values[ values < 0.0 ] = 0.0
    #     values[:zero_index] = 0.0
    #     values = values.astype(int)
    #     return list(values)

    # optimization_funcs = [
    #     optimization1,
    #     optimization2,
    #     optimization3,
    # ]
    # # Get data
    # x = np.array(index_list)
    # y = np.array(value_list)
    # # Fit curves
    # end    = int(len(index_list) * 1.25)
    # domain = np.array(range(1, end))
    # zero_index = len(value_list) - 1
    # zero_value = value_list[zero_index]
    # regressed_opts = [ curve_fit(func, x, y)[0] for func in optimization_funcs ]
    # prediction_list = [
    #     process(func, opts, domain, zero_index, zero_value)
    #     for func, opts in zip(optimization_funcs, regressed_opts)
    # ]
    # index_list = list(domain)
    prediction_list = []

    date_seen_dict.pop('UNKNOWN', None)
    bar_label_list = sorted(date_seen_dict.keys())
    bar_value_list1 = [ date_taken_dict[date][0] for date in bar_label_list ]
    bar_value_list2 = [ date_taken_dict[date][1] for date in bar_label_list ]
    bar_value_list3 = [ date_seen_dict[date][0] for date in bar_label_list ]
    bar_value_list4 = [ date_seen_dict[date][1] for date in bar_label_list ]
    bar_value_list5 = [ date_seen_dict[date][2] for date in bar_label_list ]
    bar_value_list6 = [ date_seen_dict[date][3] for date in bar_label_list ]

    # label_list += ['Models'] + [''] * (len(index_list) - len(label_list) - 1)
    # value_list += [0] * (len(index_list) - len(value_list))

    # Counts
    imgsetid_list = ibs.get_valid_imgsetids()
    gid_list = ibs.get_valid_gids()
    gid_list = filter_images_imageset(gid_list)
    aid_list = ibs.get_valid_aids()
    aid_list = filter_annots_imageset(aid_list)
    nid_list = ibs.get_valid_nids()
    nid_list = filter_names_imageset(nid_list)
    # contrib_list = ibs.get_valid_contrib_rowids()
    note_list = ibs.get_image_notes(gid_list)
    note_list = [
        ','.join(note.split(',')[:-1])
        for note in note_list
    ]
    contrib_list = set(note_list)
    # nid_list = ibs.get_valid_nids()
    aid_list_count = ibs.filter_annots_general(aid_list, filter_kw=filter_kw)
    aid_list_count = filter_annots_imageset(aid_list_count)
    gid_list_count = list(set(ibs.get_annot_gids(aid_list_count)))
    nid_list_count_dup = ibs.get_annot_name_rowids(aid_list_count)
    nid_list_count = list(set(nid_list_count_dup))

    # Calculate the Petersen-Lincoln index form the last two days
    from ibeis.other import dbinfo as dbinfo_
    try:
        try:
            raise KeyError()
            vals = dbinfo_.estimate_ggr_count(ibs)
            nsight1, nsight2, resight, pl_index, pl_error = vals
            # pl_index = 'Undefined - Zero recaptured (k = 0)'
        except KeyError:
            index1 = bar_label_list.index('2016/01/30')
            index2 = bar_label_list.index('2016/01/31')
            c1 = bar_value_list4[index1]
            c2 = bar_value_list4[index2]
            c3 = bar_value_list6[index2]
            pl_index, pl_error = dbinfo_.sight_resight_count(c1, c2, c3)
    except (IndexError, ValueError):
        pl_index = 0
        pl_error = 0

    # Get the markers
    gid_list_markers = ibs.get_annot_gids(aid_list_count)
    gps_list_markers = map(list, ibs.get_image_gps(gid_list_markers))
    gps_list_markers_all = map(list, ibs.get_image_gps(gid_list))

    REMOVE_DUP_CODE = True
    if not REMOVE_DUP_CODE:
        # Get the tracks
        nid_track_dict = ut.ddict(list)
        for nid, gps in zip(nid_list_count_dup, gps_list_markers):
            if gps[0] == -1.0 and gps[1] == -1.0:
                continue
            nid_track_dict[nid].append(gps)
        gps_list_tracks = [ nid_track_dict[nid] for nid in sorted(nid_track_dict.keys()) ]
    else:
        __nid_list, gps_track_list, aid_track_list = ibs.get_name_gps_tracks(aid_list=aid_list_count)
        gps_list_tracks = list(map(lambda x: list(map(list, x)), gps_track_list))

    gps_list_markers = [ gps for gps in gps_list_markers if tuple(gps) != (-1, -1, ) ]
    gps_list_markers_all = [ gps for gps in gps_list_markers_all if tuple(gps) != (-1, -1, ) ]
    gps_list_tracks = [
        [ gps for gps in gps_list_track if tuple(gps) != (-1, -1, ) ]
        for gps_list_track in gps_list_tracks
    ]

    valid_aids = ibs.get_valid_aids()
    valid_aids = filter_annots_imageset(valid_aids)
    used_gids = list(set( ibs.get_annot_gids(valid_aids) ))
    # used_contrib_tags = list(set( ibs.get_image_contributor_tag(used_gids) ))
    note_list = ibs.get_image_notes(used_gids)
    note_list = [
        ','.join(note.split(',')[:-1])
        for note in note_list
    ]
    used_contrib_tags = set(note_list)

    # Get Age and sex (By Annot)
    # annot_sex_list = ibs.get_annot_sex(valid_aids_)
    # annot_age_months_est_min = ibs.get_annot_age_months_est_min(valid_aids_)
    # annot_age_months_est_max = ibs.get_annot_age_months_est_max(valid_aids_)
    # age_list = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    # for sex, min_age, max_age in zip(annot_sex_list, annot_age_months_est_min, annot_age_months_est_max):
    #     if sex not in [0, 1]:
    #         sex = 2
    #         # continue
    #     if (min_age is None or min_age < 12) and max_age < 12:
    #         age_list[sex][0] += 1
    #     elif 12 <= min_age and min_age < 36 and 12 <= max_age and max_age < 36:
    #         age_list[sex][1] += 1
    #     elif 36 <= min_age and (36 <= max_age or max_age is None):
    #         age_list[sex][2] += 1

    # Get Age and sex (By Name)
    name_sex_list = ibs.get_name_sex(nid_list_count)
    name_age_months_est_mins_list = ibs.get_name_age_months_est_min(nid_list_count)
    name_age_months_est_maxs_list = ibs.get_name_age_months_est_max(nid_list_count)
    age_list = [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
    age_unreviewed = 0
    age_ambiguous = 0
    for nid, sex, min_ages, max_ages in zip(nid_list_count, name_sex_list, name_age_months_est_mins_list, name_age_months_est_maxs_list):
        if len(set(min_ages)) > 1 or len(set(max_ages)) > 1:
            # print('[web] Invalid name %r: Cannot have more than one age' % (nid, ))
            age_ambiguous += 1
            continue
        min_age = None
        max_age = None
        if len(min_ages) > 0:
            min_age = min_ages[0]
        if len(max_ages) > 0:
            max_age = max_ages[0]
        # Histogram
        if (min_age is None and max_age is None) or (min_age is -1 and max_age is -1):
            # print('[web] Unreviewded name %r: Specify the age for the name' % (nid, ))
            age_unreviewed += 1
            continue
        if sex not in [0, 1]:
            sex = 2
            # continue
        if (min_age is None or min_age < 12) and max_age < 12:
            age_list[sex][0] += 1
        elif 12 <= min_age and min_age < 24 and 12 <= max_age and max_age < 24:
            age_list[sex][1] += 1
        elif 24 <= min_age and min_age < 36 and 24 <= max_age and max_age < 36:
            age_list[sex][2] += 1
        elif 36 <= min_age and (36 <= max_age or max_age is None):
            age_list[sex][3] += 1

    age_total = sum(map(sum, age_list)) + age_unreviewed + age_ambiguous
    age_total = np.nan if age_total == 0 else age_total
    age_fmt_str = (lambda x: '% 4d (% 2.02f%%)' % (x, 100 * x / age_total, ))
    age_str_list = [
        [
            age_fmt_str(age)
            for age in age_list_
        ]
        for age_list_ in age_list
    ]
    age_str_list.append(age_fmt_str(age_unreviewed))
    age_str_list.append(age_fmt_str(age_ambiguous))

    # dbinfo_str = dbinfo()
    dbinfo_str = 'SKIPPED DBINFO'

    path_dict = ibs.compute_ggr_path_dict()
    if 'North' in path_dict:
        path_dict.pop('North')
    if 'Core' in path_dict:
        path_dict.pop('Core')

    return appf.template('view',
                         line_index_list=index_list,
                         line_label_list=label_list,
                         line_value_list=value_list,
                         prediction_list=prediction_list,
                         pl_index=pl_index,
                         pl_error=pl_error,
                         gps_list_markers=gps_list_markers,
                         gps_list_markers_all=gps_list_markers_all,
                         gps_list_tracks=gps_list_tracks,
                         path_dict=path_dict,
                         bar_label_list=bar_label_list,
                         bar_value_list1=bar_value_list1,
                         bar_value_list2=bar_value_list2,
                         bar_value_list3=bar_value_list3,
                         bar_value_list4=bar_value_list4,
                         bar_value_list5=bar_value_list5,
                         bar_value_list6=bar_value_list6,
                         age_list=age_list,
                         age_str_list=age_str_list,
                         age_ambiguous=age_ambiguous,
                         age_unreviewed=age_unreviewed,
                         age_total=age_total,
                         dbinfo_str=dbinfo_str,
                         imgsetid_list=imgsetid_list,
                         imgsetid_list_str=','.join(map(str, imgsetid_list)),
                         num_imgsetids=len(imgsetid_list),
                         gid_list=gid_list,
                         gid_list_str=','.join(map(str, gid_list)),
                         num_gids=len(gid_list),
                         contrib_list=contrib_list,
                         contrib_list_str=','.join(map(str, contrib_list)),
                         num_contribs=len(contrib_list),
                         gid_list_count=gid_list_count,
                         gid_list_count_str=','.join(map(str, gid_list_count)),
                         num_gids_count=len(gid_list_count),
                         aid_list=aid_list,
                         aid_list_str=','.join(map(str, aid_list)),
                         num_aids=len(aid_list),
                         aid_list_count=aid_list_count,
                         aid_list_count_str=','.join(map(str, aid_list_count)),
                         num_aids_count=len(aid_list_count),
                         nid_list=nid_list,
                         nid_list_str=','.join(map(str, nid_list)),
                         num_nids=len(nid_list),
                         nid_list_count=nid_list_count,
                         nid_list_count_str=','.join(map(str, nid_list_count)),
                         num_nids_count=len(nid_list_count),
                         used_gids=used_gids,
                         num_used_gids=len(used_gids),
                         used_contribs=used_contrib_tags,
                         num_used_contribs=len(used_contrib_tags),
                         __wrapper_header__=False)


@register_route('/view/imagesets/', methods=['GET'])
def view_imagesets():
    ibs = current_app.ibs
    filtered = True
    imgsetid = request.args.get('imgsetid', '')
    if len(imgsetid) > 0:
        imgsetid_list = imgsetid.strip().split(',')
        imgsetid_list = [ None if imgsetid_ == 'None' or imgsetid_ == '' else int(imgsetid_) for imgsetid_ in imgsetid_list ]
    else:
        imgsetid_list = ibs.get_valid_imgsetids()
        filtered = False
    start_time_posix_list = ibs.get_imageset_start_time_posix(imgsetid_list)
    datetime_list = [
        ut.unixtime_to_datetimestr(start_time_posix)
        if start_time_posix is not None else
        'Unknown'
        for start_time_posix in start_time_posix_list
    ]
    gids_list = [ ibs.get_valid_gids(imgsetid=imgsetid_) for imgsetid_ in imgsetid_list ]
    aids_list = [ ut.flatten(ibs.get_image_aids(gid_list)) for gid_list in gids_list ]
    images_reviewed_list           = [ appf.imageset_image_processed(ibs, gid_list) for gid_list in gids_list ]
    annots_reviewed_viewpoint_list = [ appf.imageset_annot_viewpoint_processed(ibs, aid_list) for aid_list in aids_list ]
    annots_reviewed_quality_list   = [ appf.imageset_annot_quality_processed(ibs, aid_list) for aid_list in aids_list ]
    image_processed_list           = [ images_reviewed.count(True) for images_reviewed in images_reviewed_list ]
    annot_processed_viewpoint_list = [ annots_reviewed.count(True) for annots_reviewed in annots_reviewed_viewpoint_list ]
    annot_processed_quality_list   = [ annots_reviewed.count(True) for annots_reviewed in annots_reviewed_quality_list ]
    reviewed_list = [ all(images_reviewed) and all(annots_reviewed_viewpoint) and all(annot_processed_quality) for images_reviewed, annots_reviewed_viewpoint, annot_processed_quality in zip(images_reviewed_list, annots_reviewed_viewpoint_list, annots_reviewed_quality_list) ]
    imageset_list = zip(
        imgsetid_list,
        ibs.get_imageset_text(imgsetid_list),
        ibs.get_imageset_num_gids(imgsetid_list),
        image_processed_list,
        ibs.get_imageset_num_aids(imgsetid_list),
        annot_processed_viewpoint_list,
        annot_processed_quality_list,
        start_time_posix_list,
        datetime_list,
        reviewed_list,
    )
    imageset_list.sort(key=lambda t: t[7])
    return appf.template('view', 'imagesets',
                         filtered=filtered,
                         imgsetid_list=imgsetid_list,
                         imgsetid_list_str=','.join(map(str, imgsetid_list)),
                         num_imgsetids=len(imgsetid_list),
                         imageset_list=imageset_list,
                         num_imagesets=len(imageset_list))


@register_route('/view/image/<gid>/', methods=['GET'])
def image_view_api(gid=None, thumbnail=False, fresh=False, **kwargs):
    r"""
    Returns the base64 encoded image of image <gid>

    RESTful:
        Method: GET
        URL:    /image/view/<gid>/
    """
    encoded = routes_ajax.image_src(gid, thumbnail=thumbnail, fresh=fresh, **kwargs)
    return appf.template(None, 'single', encoded=encoded)


@register_route('/view/images/', methods=['GET'])
def view_images():
    ibs = current_app.ibs
    filtered = True
    imgsetid_list = []
    gid = request.args.get('gid', '')
    imgsetid = request.args.get('imgsetid', '')
    page = max(0, int(request.args.get('page', 1)))
    if len(gid) > 0:
        gid_list = gid.strip().split(',')
        gid_list = [ None if gid_ == 'None' or gid_ == '' else int(gid_) for gid_ in gid_list ]
    elif len(imgsetid) > 0:
        imgsetid_list = imgsetid.strip().split(',')
        imgsetid_list = [ None if imgsetid_ == 'None' or imgsetid_ == '' else int(imgsetid_) for imgsetid_ in imgsetid_list ]
        gid_list = ut.flatten([ ibs.get_valid_gids(imgsetid=imgsetid) for imgsetid_ in imgsetid_list ])
    else:
        gid_list = ibs.get_valid_gids()
        filtered = False
    # Page
    page_start = min(len(gid_list), (page - 1) * appf.PAGE_SIZE)
    page_end   = min(len(gid_list), page * appf.PAGE_SIZE)
    page_total = int(math.ceil(len(gid_list) / appf.PAGE_SIZE))
    page_previous = None if page_start == 0 else page - 1
    page_next = None if page_end == len(gid_list) else page + 1
    gid_list = gid_list[page_start:page_end]
    print('[web] Loading Page [ %d -> %d ] (%d), Prev: %s, Next: %s' % (page_start, page_end, len(gid_list), page_previous, page_next, ))
    image_unixtime_list = ibs.get_image_unixtime(gid_list)
    datetime_list = [
        ut.unixtime_to_datetimestr(image_unixtime)
        if image_unixtime is not None
        else
        'Unknown'
        for image_unixtime in image_unixtime_list
    ]
    image_list = zip(
        gid_list,
        [ ','.join(map(str, imgsetid_list_)) for imgsetid_list_ in ibs.get_image_imgsetids(gid_list) ],
        ibs.get_image_gnames(gid_list),
        image_unixtime_list,
        datetime_list,
        ibs.get_image_gps(gid_list),
        ibs.get_image_party_tag(gid_list),
        ibs.get_image_contributor_tag(gid_list),
        ibs.get_image_notes(gid_list),
        appf.imageset_image_processed(ibs, gid_list),
    )
    image_list.sort(key=lambda t: t[3])
    return appf.template('view', 'images',
                         filtered=filtered,
                         imgsetid_list=imgsetid_list,
                         imgsetid_list_str=','.join(map(str, imgsetid_list)),
                         num_imgsetids=len(imgsetid_list),
                         gid_list=gid_list,
                         gid_list_str=','.join(map(str, gid_list)),
                         num_gids=len(gid_list),
                         image_list=image_list,
                         num_images=len(image_list),
                         page=page,
                         page_start=page_start,
                         page_end=page_end,
                         page_total=page_total,
                         page_previous=page_previous,
                         page_next=page_next)


@register_route('/view/annotations/', methods=['GET'])
def view_annotations():
    ibs = current_app.ibs
    filtered = True
    imgsetid_list = []
    gid_list = []
    aid = request.args.get('aid', '')
    gid = request.args.get('gid', '')
    imgsetid = request.args.get('imgsetid', '')
    page = max(0, int(request.args.get('page', 1)))
    if len(aid) > 0:
        aid_list = aid.strip().split(',')
        aid_list = [ None if aid_ == 'None' or aid_ == '' else int(aid_) for aid_ in aid_list ]
    elif len(gid) > 0:
        gid_list = gid.strip().split(',')
        gid_list = [ None if gid_ == 'None' or gid_ == '' else int(gid_) for gid_ in gid_list ]
        aid_list = ut.flatten(ibs.get_image_aids(gid_list))
    elif len(imgsetid) > 0:
        imgsetid_list = imgsetid.strip().split(',')
        imgsetid_list = [ None if imgsetid_ == 'None' or imgsetid_ == '' else int(imgsetid_) for imgsetid_ in imgsetid_list ]
        gid_list = ut.flatten([ ibs.get_valid_gids(imgsetid=imgsetid_) for imgsetid_ in imgsetid_list ])
        aid_list = ut.flatten(ibs.get_image_aids(gid_list))
    else:
        aid_list = ibs.get_valid_aids()
        filtered = False
    # Page
    page_start = min(len(aid_list), (page - 1) * appf.PAGE_SIZE)
    page_end   = min(len(aid_list), page * appf.PAGE_SIZE)
    page_total = int(math.ceil(len(aid_list) / appf.PAGE_SIZE))
    page_previous = None if page_start == 0 else page - 1
    page_next = None if page_end == len(aid_list) else page + 1
    aid_list = aid_list[page_start:page_end]
    print('[web] Loading Page [ %d -> %d ] (%d), Prev: %s, Next: %s' % (page_start, page_end, len(aid_list), page_previous, page_next, ))
    annotation_list = zip(
        aid_list,
        ibs.get_annot_gids(aid_list),
        [ ','.join(map(str, imgsetid_list_)) for imgsetid_list_ in ibs.get_annot_imgsetids(aid_list) ],
        ibs.get_annot_image_names(aid_list),
        ibs.get_annot_names(aid_list),
        ibs.get_annot_exemplar_flags(aid_list),
        ibs.get_annot_species_texts(aid_list),
        ibs.get_annot_yaw_texts(aid_list),
        ibs.get_annot_quality_texts(aid_list),
        ibs.get_annot_sex_texts(aid_list),
        ibs.get_annot_age_months_est(aid_list),
        ibs.get_annot_reviewed(aid_list),
        # [ reviewed_viewpoint and reviewed_quality for reviewed_viewpoint, reviewed_quality in zip(appf.imageset_annot_viewpoint_processed(ibs, aid_list), appf.imageset_annot_quality_processed(ibs, aid_list)) ],
    )
    annotation_list.sort(key=lambda t: t[0])
    return appf.template('view', 'annotations',
                         filtered=filtered,
                         imgsetid_list=imgsetid_list,
                         imgsetid_list_str=','.join(map(str, imgsetid_list)),
                         num_imgsetids=len(imgsetid_list),
                         gid_list=gid_list,
                         gid_list_str=','.join(map(str, gid_list)),
                         num_gids=len(gid_list),
                         aid_list=aid_list,
                         aid_list_str=','.join(map(str, aid_list)),
                         num_aids=len(aid_list),
                         annotation_list=annotation_list,
                         num_annotations=len(annotation_list),
                         page=page,
                         page_start=page_start,
                         page_end=page_end,
                         page_total=page_total,
                         page_previous=page_previous,
                         page_next=page_next)


@register_route('/view/names/', methods=['GET'])
def view_names():
    ibs = current_app.ibs
    filtered = True
    aid_list = []
    imgsetid_list = []
    gid_list = []
    nid = request.args.get('nid', '')
    aid = request.args.get('aid', '')
    gid = request.args.get('gid', '')
    imgsetid = request.args.get('imgsetid', '')
    page = max(0, int(request.args.get('page', 1)))
    if len(nid) > 0:
        nid_list = nid.strip().split(',')
        nid_list = [ None if nid_ == 'None' or nid_ == '' else int(nid_) for nid_ in nid_list ]
    if len(aid) > 0:
        aid_list = aid.strip().split(',')
        aid_list = [ None if aid_ == 'None' or aid_ == '' else int(aid_) for aid_ in aid_list ]
        nid_list = ibs.get_annot_name_rowids(aid_list)
    elif len(gid) > 0:
        gid_list = gid.strip().split(',')
        gid_list = [ None if gid_ == 'None' or gid_ == '' else int(gid_) for gid_ in gid_list ]
        aid_list = ut.flatten(ibs.get_image_aids(gid_list))
        nid_list = ibs.get_annot_name_rowids(aid_list)
    elif len(imgsetid) > 0:
        imgsetid_list = imgsetid.strip().split(',')
        imgsetid_list = [ None if imgsetid_ == 'None' or imgsetid_ == '' else int(imgsetid_) for imgsetid_ in imgsetid_list ]
        gid_list = ut.flatten([ ibs.get_valid_gids(imgsetid=imgsetid_) for imgsetid_ in imgsetid_list ])
        aid_list = ut.flatten(ibs.get_image_aids(gid_list))
        nid_list = ibs.get_annot_name_rowids(aid_list)
    else:
        nid_list = ibs.get_valid_nids()
        filtered = False
    # Page
    appf.PAGE_SIZE_ = int(appf.PAGE_SIZE / 5)
    page_start = min(len(nid_list), (page - 1) * appf.PAGE_SIZE_)
    page_end   = min(len(nid_list), page * appf.PAGE_SIZE_)
    page_total = int(math.ceil(len(nid_list) / appf.PAGE_SIZE_))
    page_previous = None if page_start == 0 else page - 1
    page_next = None if page_end == len(nid_list) else page + 1
    nid_list = nid_list[page_start:page_end]
    print('[web] Loading Page [ %d -> %d ] (%d), Prev: %s, Next: %s' % (page_start, page_end, len(nid_list), page_previous, page_next, ))
    aids_list = ibs.get_name_aids(nid_list)
    annotations_list = [ zip(
        aid_list_,
        ibs.get_annot_gids(aid_list_),
        [ ','.join(map(str, imgsetid_list_)) for imgsetid_list_ in ibs.get_annot_imgsetids(aid_list_) ],
        ibs.get_annot_image_names(aid_list_),
        ibs.get_annot_names(aid_list_),
        ibs.get_annot_exemplar_flags(aid_list_),
        ibs.get_annot_species_texts(aid_list_),
        ibs.get_annot_yaw_texts(aid_list_),
        ibs.get_annot_quality_texts(aid_list_),
        ibs.get_annot_sex_texts(aid_list_),
        ibs.get_annot_age_months_est(aid_list_),
        [ reviewed_viewpoint and reviewed_quality for reviewed_viewpoint, reviewed_quality in zip(appf.imageset_annot_viewpoint_processed(ibs, aid_list_), appf.imageset_annot_quality_processed(ibs, aid_list_)) ],
    ) for aid_list_ in aids_list ]
    name_list = zip(
        nid_list,
        annotations_list
    )
    name_list.sort(key=lambda t: t[0])
    return appf.template('view', 'names',
                         filtered=filtered,
                         imgsetid_list=imgsetid_list,
                         imgsetid_list_str=','.join(map(str, imgsetid_list)),
                         num_imgsetids=len(imgsetid_list),
                         gid_list=gid_list,
                         gid_list_str=','.join(map(str, gid_list)),
                         num_gids=len(gid_list),
                         aid_list=aid_list,
                         aid_list_str=','.join(map(str, aid_list)),
                         num_aids=len(aid_list),
                         nid_list=nid_list,
                         nid_list_str=','.join(map(str, nid_list)),
                         num_nids=len(nid_list),
                         name_list=name_list,
                         num_names=len(name_list),
                         page=page,
                         page_start=page_start,
                         page_end=page_end,
                         page_total=page_total,
                         page_previous=page_previous,
                         page_next=page_next)


@register_route('/turk/', methods=['GET'])
def turk():
    imgsetid = request.args.get('imgsetid', '')
    imgsetid = None if imgsetid == 'None' or imgsetid == '' else int(imgsetid)
    return appf.template('turk', None, imgsetid=imgsetid)


def _make_review_image_info(ibs, gid):
    """
    Example:
        >>> # DISABLE_DOCTEST
        >>> from ibeis.web.apis_detect import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb(defaultdb='testdb1')
        >>> gid = ibs.get_valid_gids()[0]
    """
    # Shows how to use new object-like interface to populate data
    import numpy as np
    image = ibs.images([gid])[0]
    annots = image.annots
    width, height = image.sizes
    bbox_denom = np.array([width, height, width, height])
    annotation_list = []
    for aid in annots.aids:
        annot_ = ibs.annots(aid)[0]
        bbox = np.array(annot_.bboxes)
        bbox_percent = bbox / bbox_denom * 100
        temp = {
            'left'   :  bbox_percent[0],
            'top'    :  bbox_percent[1],
            'width'  :  bbox_percent[2],
            'height' :  bbox_percent[3],
            'label'  :  annot_.species,
            'id'     :  annot_.aids,
            'theta'  :  annot_.thetas,
            'tags'   :  annot_.case_tags,
        }
    annotation_list.append(temp)


@register_route('/turk/detection/', methods=['GET'])
def turk_detection():

    ibs = current_app.ibs
    refer_aid = request.args.get('refer_aid', None)
    imgsetid = request.args.get('imgsetid', '')
    imgsetid = None if imgsetid == 'None' or imgsetid == '' else int(imgsetid)

    gid_list = ibs.get_valid_gids(imgsetid=imgsetid)
    reviewed_list = appf.imageset_image_processed(ibs, gid_list)
    progress = '%0.2f' % (100.0 * reviewed_list.count(True) / len(gid_list), )

    imagesettext = None if imgsetid is None else ibs.get_imageset_text(imgsetid)
    gid = request.args.get('gid', '')
    if len(gid) > 0:
        gid = int(gid)
    else:
        gid_list_ = ut.filterfalse_items(gid_list, reviewed_list)
        if len(gid_list_) == 0:
            gid = None
        else:
            # gid = gid_list_[0]
            gid = random.choice(gid_list_)
    previous = request.args.get('previous', None)
    finished = gid is None
    review = 'review' in request.args.keys()
    display_instructions = request.cookies.get('ia-detection_instructions_seen', 1) == 0
    display_species_examples = False  # request.cookies.get('ia-detection_example_species_seen', 0) == 0
    if not finished:
        gpath = ibs.get_image_thumbpath(gid, ensure_paths=True, draw_annots=False)
        imgdata = ibs.get_image_imgdata(gid)
        image_src = appf.embed_image_html(imgdata)
        # Get annotations
        width, height = ibs.get_image_sizes(gid)
        aid_list = ibs.get_image_aids(gid)
        annot_bbox_list = ibs.get_annot_bboxes(aid_list)
        annot_thetas_list = ibs.get_annot_thetas(aid_list)
        species_list = ibs.get_annot_species_texts(aid_list)
        # Get annotation bounding boxes
        annotation_list = []
        for aid, annot_bbox, annot_theta, species in zip(aid_list, annot_bbox_list, annot_thetas_list, species_list):
            temp = {}
            temp['left']   = 100.0 * (annot_bbox[0] / width)
            temp['top']    = 100.0 * (annot_bbox[1] / height)
            temp['width']  = 100.0 * (annot_bbox[2] / width)
            temp['height'] = 100.0 * (annot_bbox[3] / height)
            temp['label']  = species
            temp['id']     = aid
            temp['theta']  = float(annot_theta)
            annotation_list.append(temp)
        if len(species_list) > 0:
            species = max(set(species_list), key=species_list.count)  # Get most common species
        elif appf.default_species(ibs) is not None:
            species = appf.default_species(ibs)
        else:
            species = KEY_DEFAULTS[SPECIES_KEY]
    else:
        gpath = None
        species = None
        image_src = None
        annotation_list = []
    callback_url = '%s?imgsetid=%s' % (url_for('submit_detection'), imgsetid, )
    return appf.template('turk', 'detection',
                         imgsetid=imgsetid,
                         gid=gid,
                         refer_aid=refer_aid,
                         species=species,
                         image_path=gpath,
                         image_src=image_src,
                         previous=previous,
                         imagesettext=imagesettext,
                         progress=progress,
                         finished=finished,
                         annotation_list=annotation_list,
                         display_instructions=display_instructions,
                         display_species_examples=display_species_examples,
                         callback_url=callback_url,
                         callback_method='POST',
                         EMBEDDED_CSS=None,
                         EMBEDDED_JAVASCRIPT=None,
                         review=review)


@register_route('/turk/detection/dynamic/', methods=['GET'])
def turk_detection_dynamic():
    ibs = current_app.ibs
    gid = request.args.get('gid', None)

    gpath = ibs.get_image_thumbpath(gid, ensure_paths=True, draw_annots=False)
    image = ibs.get_image_imgdata(gid)
    image_src = appf.embed_image_html(image)
    # Get annotations
    width, height = ibs.get_image_sizes(gid)
    aid_list = ibs.get_image_aids(gid)
    annot_bbox_list = ibs.get_annot_bboxes(aid_list)
    annot_thetas_list = ibs.get_annot_thetas(aid_list)
    species_list = ibs.get_annot_species_texts(aid_list)
    # Get annotation bounding boxes
    annotation_list = []
    for aid, annot_bbox, annot_theta, species in zip(aid_list, annot_bbox_list, annot_thetas_list, species_list):
        temp = {}
        temp['left']   = 100.0 * (annot_bbox[0] / width)
        temp['top']    = 100.0 * (annot_bbox[1] / height)
        temp['width']  = 100.0 * (annot_bbox[2] / width)
        temp['height'] = 100.0 * (annot_bbox[3] / height)
        temp['label']  = species
        temp['id']     = aid
        temp['theta']  = float(annot_theta)
        annotation_list.append(temp)
    if len(species_list) > 0:
        species = max(set(species_list), key=species_list.count)  # Get most common species
    elif appf.default_species(ibs) is not None:
        species = appf.default_species(ibs)
    else:
        species = KEY_DEFAULTS[SPECIES_KEY]

    callback_url = '%s?imgsetid=%s' % (url_for('submit_detection'), gid, )
    return appf.template('turk', 'detection_dynamic',
                         gid=gid,
                         refer_aid=None,
                         species=species,
                         image_path=gpath,
                         image_src=image_src,
                         annotation_list=annotation_list,
                         callback_url=callback_url,
                         callback_method='POST',
                         EMBEDDED_CSS=None,
                         EMBEDDED_JAVASCRIPT=None,
                         __wrapper__=False)


@register_route('/turk/annotation/', methods=['GET'])
def turk_annotation():
    """
    CommandLine:
        python -m ibeis.web.app --exec-turk_viewpoint --db PZ_Master1

    Example:
        >>> # SCRIPT
        >>> from ibeis.other.ibsfuncs import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb(defaultdb='PZ_Master1')
        >>> aid_list_ = ibs.find_unlabeled_name_members(suspect_yaws=True)
        >>> aid_list = ibs.filter_aids_to_quality(aid_list_, 'good', unknown_ok=False)
        >>> ibs.start_web_annot_groupreview(aid_list)
    """
    ibs = current_app.ibs
    tup = appf.get_turk_annot_args(appf.imageset_annot_processed)
    (aid_list, reviewed_list, imgsetid, src_ag, dst_ag, progress, aid, previous) = tup

    review = 'review' in request.args.keys()
    finished = aid is None
    display_instructions = request.cookies.get('ia-annotation_instructions_seen', 1) == 0
    if not finished:
        gid       = ibs.get_annot_gids(aid)
        gpath     = ibs.get_annot_chip_fpath(aid)
        image     = vt.imread(gpath)
        image_src = appf.embed_image_html(image)
        # image_src = routes_ajax.annotation_src(aid)
        species   = ibs.get_annot_species_texts(aid)
        viewpoint_value = appf.convert_yaw_to_old_viewpoint(ibs.get_annot_yaws(aid))
        quality_value = ibs.get_annot_qualities(aid)
        if quality_value in [-1, None]:
            quality_value = None
        elif quality_value > 2:
            quality_value = 2
        elif quality_value <= 2:
            quality_value = 1
        multiple_value = ibs.get_annot_multiple(aid) == 1
    else:
        gid       = None
        gpath     = None
        image_src = None
        species   = None
        viewpoint_value = None
        quality_value = None
        multiple_value = None

    imagesettext = ibs.get_imageset_text(imgsetid)

    species_rowids = ibs._get_all_species_rowids()
    species_nice_list = ibs.get_species_nice(species_rowids)

    combined_list = sorted(zip(species_nice_list, species_rowids))
    species_nice_list = [ combined[0] for combined in combined_list ]
    species_rowids = [ combined[1] for combined in combined_list ]

    species_text_list = ibs.get_species_texts(species_rowids)
    species_selected_list = [ species == species_ for species_ in species_text_list ]
    species_list = zip(species_nice_list, species_text_list, species_selected_list)
    species_list = [ ('Unspecified', const.UNKNOWN, True) ] + species_list

    callback_url = url_for('submit_annotation')
    return appf.template('turk', 'annotation',
                         imgsetid=imgsetid,
                         src_ag=src_ag,
                         dst_ag=dst_ag,
                         gid=gid,
                         aid=aid,
                         viewpoint_value=viewpoint_value,
                         quality_value=quality_value,
                         multiple_value=multiple_value,
                         image_path=gpath,
                         image_src=image_src,
                         previous=previous,
                         species_list=species_list,
                         imagesettext=imagesettext,
                         progress=progress,
                         finished=finished,
                         display_instructions=display_instructions,
                         callback_url=callback_url,
                         callback_method='POST',
                         EMBEDDED_CSS=None,
                         EMBEDDED_JAVASCRIPT=None,
                         review=review)


@register_route('/turk/annotation/dynamic/', methods=['GET'])
def turk_annotation_dynamic():
    ibs = current_app.ibs
    aid = request.args.get('aid', None)
    imgsetid = request.args.get('imgsetid', None)

    review = 'review' in request.args.keys()
    gid       = ibs.get_annot_gids(aid)
    gpath     = ibs.get_annot_chip_fpath(aid)
    image     = vt.imread(gpath)
    image_src = appf.embed_image_html(image)
    species   = ibs.get_annot_species_texts(aid)
    viewpoint_value = appf.convert_yaw_to_old_viewpoint(ibs.get_annot_yaws(aid))
    quality_value = ibs.get_annot_qualities(aid)
    if quality_value == -1:
        quality_value = None
    if quality_value == 0:
        quality_value = 1

    species_rowids = ibs._get_all_species_rowids()
    species_nice_list = ibs.get_species_nice(species_rowids)

    combined_list = sorted(zip(species_nice_list, species_rowids))
    species_nice_list = [ combined[0] for combined in combined_list ]
    species_rowids = [ combined[1] for combined in combined_list ]

    species_text_list = ibs.get_species_texts(species_rowids)
    species_selected_list = [ species == species_ for species_ in species_text_list ]
    species_list = zip(species_nice_list, species_text_list, species_selected_list)
    species_list = [ ('Unspecified', const.UNKNOWN, True) ] + species_list

    callback_url = url_for('submit_annotation')
    return appf.template('turk', 'annotation_dynamic',
                         imgsetid=imgsetid,
                         gid=gid,
                         aid=aid,
                         viewpoint_value=viewpoint_value,
                         quality_value=quality_value,
                         image_path=gpath,
                         image_src=image_src,
                         species_list=species_list,
                         callback_url=callback_url,
                         callback_method='POST',
                         EMBEDDED_CSS=None,
                         EMBEDDED_JAVASCRIPT=None,
                         review=review,
                         __wrapper__=False)


@register_route('/turk/viewpoint/', methods=['GET'])
def turk_viewpoint():
    """
    CommandLine:
        python -m ibeis.web.app --exec-turk_viewpoint --db PZ_Master1

    Example:
        >>> # SCRIPT
        >>> from ibeis.other.ibsfuncs import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb(defaultdb='PZ_Master1')
        >>> aid_list_ = ibs.find_unlabeled_name_members(suspect_yaws=True)
        >>> aid_list = ibs.filter_aids_to_quality(aid_list_, 'good', unknown_ok=False)
        >>> ibs.start_web_annot_groupreview(aid_list)
    """
    ibs = current_app.ibs
    tup = appf.get_turk_annot_args(appf.imageset_annot_viewpoint_processed)
    (aid_list, reviewed_list, imgsetid, src_ag, dst_ag, progress, aid, previous) = tup

    value = appf.convert_yaw_to_old_viewpoint(ibs.get_annot_yaws(aid))
    review = 'review' in request.args.keys()
    finished = aid is None
    display_instructions = request.cookies.get('ia-viewpoint_instructions_seen', 1) == 0
    if not finished:
        gid       = ibs.get_annot_gids(aid)
        gpath     = ibs.get_annot_chip_fpath(aid)
        image     = vt.imread(gpath)
        image_src = appf.embed_image_html(image)
        species   = ibs.get_annot_species_texts(aid)
    else:
        gid       = None
        gpath     = None
        image_src = None
        species   = None

    imagesettext = ibs.get_imageset_text(imgsetid)

    species_rowids = ibs._get_all_species_rowids()
    species_nice_list = ibs.get_species_nice(species_rowids)

    combined_list = sorted(zip(species_nice_list, species_rowids))
    species_nice_list = [ combined[0] for combined in combined_list ]
    species_rowids = [ combined[1] for combined in combined_list ]

    species_text_list = ibs.get_species_texts(species_rowids)
    species_selected_list = [ species == species_ for species_ in species_text_list ]
    species_list = zip(species_nice_list, species_text_list, species_selected_list)
    species_list = [ ('Unspecified', const.UNKNOWN, True) ] + species_list

    return appf.template('turk', 'viewpoint',
                         imgsetid=imgsetid,
                         src_ag=src_ag,
                         dst_ag=dst_ag,
                         gid=gid,
                         aid=aid,
                         value=value,
                         image_path=gpath,
                         image_src=image_src,
                         previous=previous,
                         species_list=species_list,
                         imagesettext=imagesettext,
                         progress=progress,
                         finished=finished,
                         display_instructions=display_instructions,
                         review=review)


@register_route('/turk/quality/', methods=['GET'])
def turk_quality():
    """
    PZ Needs Tags:
        17242
        14468
        14427
        15946
        14771
        14084
        4102
        6074
        3409

    GZ Needs Tags;
    1302

    CommandLine:
        python -m ibeis.web.app --exec-turk_quality --db PZ_Master1
        python -m ibeis.web.app --exec-turk_quality --db GZ_Master1
        python -m ibeis.web.app --exec-turk_quality --db GIRM_Master1

    Example:
        >>> # SCRIPT
        >>> from ibeis.other.ibsfuncs import *  # NOQA
        >>> import ibeis
        >>> ibs = ibeis.opendb(defaultdb='testdb1')
        >>> aid_list_ = ibs.find_unlabeled_name_members(qual=True)
        >>> valid_views = ['primary', 'primary1', 'primary-1']
        >>> aid_list = ibs.filter_aids_to_viewpoint(aid_list_, valid_views, unknown_ok=False)
        >>> ibs.start_web_annot_groupreview(aid_list)
    """
    ibs = current_app.ibs
    tup = appf.get_turk_annot_args(appf.imageset_annot_quality_processed)
    (aid_list, reviewed_list, imgsetid, src_ag, dst_ag, progress, aid, previous) = tup

    value = ibs.get_annot_qualities(aid)
    if value == -1:
        value = None
    if value == 0:
        value = 1
    review = 'review' in request.args.keys()
    finished = aid is None
    display_instructions = request.cookies.get('ia-quality_instructions_seen', 1) == 0
    if not finished:
        gid       = ibs.get_annot_gids(aid)
        gpath     = ibs.get_annot_chip_fpath(aid)
        image     = vt.imread(gpath)
        image_src = appf.embed_image_html(image)
    else:
        gid       = None
        gpath     = None
        image_src = None
    imagesettext = ibs.get_imageset_text(imgsetid)
    return appf.template('turk', 'quality',
                         imgsetid=imgsetid,
                         src_ag=src_ag,
                         dst_ag=dst_ag,
                         gid=gid,
                         aid=aid,
                         value=value,
                         image_path=gpath,
                         image_src=image_src,
                         previous=previous,
                         imagesettext=imagesettext,
                         progress=progress,
                         finished=finished,
                         display_instructions=display_instructions,
                         review=review)


@register_route('/turk/additional/', methods=['GET'])
def turk_additional():
    ibs = current_app.ibs
    imgsetid = request.args.get('imgsetid', '')
    imgsetid = None if imgsetid == 'None' or imgsetid == '' else int(imgsetid)

    gid_list = ibs.get_valid_gids(imgsetid=imgsetid)
    aid_list = ut.flatten(ibs.get_image_aids(gid_list))
    nid_list = ibs.get_annot_nids(aid_list)
    reviewed_list = appf.imageset_annot_additional_processed(ibs, aid_list, nid_list)
    try:
        progress = '%0.2f' % (100.0 * reviewed_list.count(True) / len(aid_list), )
    except ZeroDivisionError:
        progress = '0.00'

    imagesettext = None if imgsetid is None else ibs.get_imageset_text(imgsetid)
    aid = request.args.get('aid', '')
    if len(aid) > 0:
        aid = int(aid)
    else:
        aid_list_ = ut.filterfalse_items(aid_list, reviewed_list)
        if len(aid_list_) == 0:
            aid = None
        else:
            # aid = aid_list_[0]
            aid = random.choice(aid_list_)
    previous = request.args.get('previous', None)
    value_sex = ibs.get_annot_sex([aid])[0]
    if value_sex >= 0:
        value_sex += 2
    else:
        value_sex = None
    value_age_min, value_age_max = ibs.get_annot_age_months_est([aid])[0]
    value_age = None
    if (value_age_min is -1 or value_age_min is None) and (value_age_max is -1 or value_age_max is None):
        value_age = 1
    if (value_age_min is 0 or value_age_min is None) and value_age_max == 2:
        value_age = 2
    elif value_age_min is 3 and value_age_max == 5:
        value_age = 3
    elif value_age_min is 6 and value_age_max == 11:
        value_age = 4
    elif value_age_min is 12 and value_age_max == 23:
        value_age = 5
    elif value_age_min is 24 and value_age_max == 35:
        value_age = 6
    elif value_age_min is 36 and (value_age_max > 36 or value_age_max is None):
        value_age = 7

    review = 'review' in request.args.keys()
    finished = aid is None
    display_instructions = request.cookies.get('ia-additional_instructions_seen', 1) == 0
    if not finished:
        gid       = ibs.get_annot_gids(aid)
        gpath     = ibs.get_annot_chip_fpath(aid)
        image     = vt.imread(gpath)
        image_src = appf.embed_image_html(image)
    else:
        gid       = None
        gpath     = None
        image_src = None
    name_aid_list = None
    nid = ibs.get_annot_name_rowids(aid)
    if nid is not None:
        name_aid_list = ibs.get_name_aids(nid)
        quality_list = ibs.get_annot_qualities(name_aid_list)
        quality_text_list = ibs.get_annot_quality_texts(name_aid_list)
        yaw_text_list = ibs.get_annot_yaw_texts(name_aid_list)
        name_aid_combined_list = list(zip(
            name_aid_list,
            quality_list,
            quality_text_list,
            yaw_text_list,
        ))
        name_aid_combined_list.sort(key=lambda t: t[1], reverse=True)
    else:
        name_aid_combined_list = []

    region_str = 'UNKNOWN'
    if aid is not None and gid is not None:
        imgsetid_list = ibs.get_image_imgsetids(gid)
        imgset_text_list = ibs.get_imageset_text(imgsetid_list)
        imgset_text_list = [
            imgset_text
            for imgset_text in imgset_text_list
            if 'GGR Special Zone' in imgset_text
        ]
        assert len(imgset_text_list) < 2
        if len(imgset_text_list) == 1:
            region_str = imgset_text_list[0]

    return appf.template('turk', 'additional',
                         imgsetid=imgsetid,
                         gid=gid,
                         aid=aid,
                         region_str=region_str,
                         value_sex=value_sex,
                         value_age=value_age,
                         image_path=gpath,
                         name_aid_combined_list=name_aid_combined_list,
                         image_src=image_src,
                         previous=previous,
                         imagesettext=imagesettext,
                         progress=progress,
                         finished=finished,
                         display_instructions=display_instructions,
                         review=review)


@register_route('/group_review/', methods=['GET'])
def group_review():
    prefill = request.args.get('prefill', '')
    if len(prefill) > 0:
        ibs = current_app.ibs
        aid_list = ibs.get_valid_aids()
        bad_species_list, bad_viewpoint_list = ibs.validate_annot_species_viewpoint_cnn(aid_list)

        GROUP_BY_PREDICTION = True
        if GROUP_BY_PREDICTION:
            grouped_dict = ut.group_items(bad_viewpoint_list, ut.get_list_column(bad_viewpoint_list, 3))
            grouped_list = grouped_dict.values()
            regrouped_items = ut.flatten(ut.sortedby(grouped_list, map(len, grouped_list)))
            candidate_aid_list = ut.get_list_column(regrouped_items, 0)
        else:
            candidate_aid_list = [ bad_viewpoint[0] for bad_viewpoint in bad_viewpoint_list]
    elif request.args.get('aid_list', None) is not None:
        aid_list = request.args.get('aid_list', '')
        if len(aid_list) > 0:
            aid_list = aid_list.replace('[', '')
            aid_list = aid_list.replace(']', '')
            aid_list = aid_list.strip().split(',')
            candidate_aid_list = [ int(aid_.strip()) for aid_ in aid_list ]
        else:
            candidate_aid_list = ''
    else:
        candidate_aid_list = ''

    return appf.template(None, 'group_review', candidate_aid_list=candidate_aid_list, mode_list=appf.VALID_TURK_MODES)


@register_route('/sightings/', methods=['GET'])
def sightings(html_encode=True):
    ibs = current_app.ibs
    complete = request.args.get('complete', None) is not None
    sightings = ibs.report_sightings_str(complete=complete, include_images=True)
    if html_encode:
        sightings = sightings.replace('\n', '<br/>')
    return sightings


@register_route('/api/', methods=['GET'], __api_prefix_check__=False)
def api_root():
    rules = current_app.url_map.iter_rules()
    rule_dict = {}
    for rule in rules:
        methods = rule.methods
        url = str(rule)
        if '/api/' in url:
            methods -= set(['HEAD', 'OPTIONS'])
            if len(methods) == 0:
                continue
            if len(methods) > 1:
                print('methods = %r' % (methods,))
            method = list(methods)[0]
            if method not in rule_dict.keys():
                rule_dict[method] = []
            rule_dict[method].append((method, url, ))
    for method in rule_dict.keys():
        rule_dict[method].sort()
    url = '%s/api/core/dbname/' % (current_app.server_url, )
    app_auth = controller_inject.get_url_authorization(url)
    return appf.template(None, 'api',
                         app_url=url,
                         app_name=controller_inject.GLOBAL_APP_NAME,
                         app_secret=controller_inject.GLOBAL_APP_SECRET,
                         app_auth=app_auth,
                         rule_list=rule_dict)


@register_route('/upload/', methods=['GET'])
def upload():
    return appf.template(None, 'upload')


@register_route('/dbinfo/', methods=['GET'])
def dbinfo():
    try:
        ibs = current_app.ibs
        dbinfo_str = ibs.get_dbinfo_str()
    except:
        dbinfo_str = ''
    dbinfo_str_formatted = '<pre>%s</pre>' % (dbinfo_str, )
    return dbinfo_str_formatted


@register_route('/counts/', methods=['GET'])
def wb_counts():
    fmt_str = '''<p># Annotations: <b>%d</b></p>
<p># MediaAssets (images): <b>%d</b></p>
<p># MarkedIndividuals: <b>%d</b></p>
<p># Encounters: <b>%d</b></p>
<p># Occurrences: <b>%d</b></p>'''

    try:
        ibs = current_app.ibs

        aid_list = ibs.get_valid_aids()
        nid_list = ibs.get_annot_nids(aid_list)
        nid_list = [ nid for nid in nid_list if nid > 0 ]
        gid_list = ibs.get_annot_gids(aid_list)
        imgset_id_list = ibs.get_valid_imgsetids()
        aids_list = ibs.get_imageset_aids(imgset_id_list)
        imgset_id_list = [
            imgset_id
            for imgset_id, aid_list_ in zip(imgset_id_list, aids_list)
            if len(aid_list_) > 0
        ]

        valid_nid_list = list(set(nid_list))
        valid_aid_list = list(set(aid_list))
        valid_gid_list = list(set(gid_list))
        valid_imgset_id_list = list(set(imgset_id_list))
        valid_imgset_id_list = list(set(imgset_id_list))

        aids_list = ibs.get_imageset_aids(valid_imgset_id_list)
        nids_list = map(ibs.get_annot_nids, aids_list)
        nids_list = map(set, nids_list)
        nids_list = ut.flatten(nids_list)

        num_nid = len(valid_nid_list)
        num_aid = len(valid_aid_list)
        num_gid = len(valid_gid_list)
        num_imgset = len(valid_imgset_id_list)
        num_encounters = len(nids_list)

        args = (num_aid, num_gid, num_nid, num_encounters, num_imgset, )
        counts_str = fmt_str % args
    except:
        counts_str = ''
    return counts_str


@register_route('/test/counts.jsp', methods=['GET'], __api_postfix_check__=False)
def wb_counts_alias1():
    return wb_counts()


@register_route('/gzgc/counts.jsp', methods=['GET'], __api_postfix_check__=False)
def wb_counts_alias2():
    return wb_counts()


@register_route('/404/', methods=['GET'])
def error404(exception=None):
    import traceback
    exception_str = str(exception)
    traceback_str = str(traceback.format_exc())
    print('[web] %r' % (exception_str, ))
    print('[web] %r' % (traceback_str, ))
    return appf.template(None, '404', exception_str=exception_str,
                         traceback_str=traceback_str)


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
