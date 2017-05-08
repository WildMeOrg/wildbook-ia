# -*- coding: utf-8 -*-
"""
Dependencies: flask, tornado
"""
from __future__ import absolute_import, division, print_function
import random
import math
import simplejson as json
from flask import request, redirect, current_app, url_for
from ibeis.control import controller_inject
from ibeis import constants as const
from ibeis.constants import KEY_DEFAULTS, SPECIES_KEY
from ibeis.web import appfuncs as appf
from ibeis.web import routes_ajax
import utool as ut
import vtool as vt
import numpy as np


CLASS_INJECT_KEY, register_ibs_method = (
    controller_inject.make_ibs_register_decorator(__name__))
register_route = controller_inject.get_ibeis_flask_route(__name__)


GLOBAL_FEEDBACK_LIMIT = 50
GLOBAL_FEEDBACK_BUFFER = []
GLOBAL_FEEDBACK_CONFIG_DICT = {
    'ranks_top': 3,
    'ranks_bot': 2,

    'score_thresh': None,
    'max_num': None,

    'filter_reviewed': True,
    'filter_photobombs': False,

    'filter_true_matches': True,
    'filter_false_matches': False,

    'filter_nonmatch_between_ccs': True,
    'filter_dup_namepairs': True,
}


@register_route('/', methods=['GET'])
def root(**kwargs):
    ibs = current_app.ibs

    dbname = ibs.dbname
    dbdir = ibs.dbdir

    embedded = dict(globals(), **locals())
    return appf.template(None, **embedded)


@register_route('/view/', methods=['GET'])
def view(**kwargs):
    ibs = current_app.ibs

    ibs.update_all_image_special_imageset()
    imgsetids_list = ibs.get_valid_imgsetids()
    gid_list = ibs.get_valid_gids()
    aid_list = ibs.get_valid_aids()
    nid_list = ibs.get_valid_nids()

    return appf.template('view',
                         num_imgsetids=len(imgsetids_list),
                         num_gids=len(gid_list),
                         num_aids=len(aid_list),
                         num_nids=len(nid_list))


@register_route('/view/viewpoints/', methods=['GET'])
def view_viewpoints(**kwargs):
    ibs = current_app.ibs

    aid_list = ibs.get_valid_aids()
    species_list = ibs.get_annot_species_texts(aid_list)
    viewpoint_list = ibs.get_annot_viewpoints(aid_list)

    species_tag_list = sorted(list(set(species_list)))
    species_rowid_list = ibs.get_species_rowids_from_text(species_tag_list)
    species_set = set(species_tag_list)
    species_nice_dict = {
        species_tag : ibs.get_species_nice(species_rowid)
        for species_tag, species_rowid in zip(species_tag_list, species_rowid_list)
    }

    viewpoint_dict = {}
    for species, viewpoint in zip(species_list, viewpoint_list):
        if species in species_set:
            if species not in viewpoint_dict:
                viewpoint_dict[species] = {}
            if viewpoint not in viewpoint_dict[species]:
                viewpoint_dict[species][viewpoint] = 0
            viewpoint_dict[species][viewpoint] += 1

    viewpoint_tag_list = const.VIEWTEXT_TO_VIEWPOINT_RADIANS.keys()
    pie_label_list = [ str(const.VIEWPOINTALIAS_NICE[_]) for _ in viewpoint_tag_list ]
    pie_values_list = [
        (
            species_nice_dict[species],
            [viewpoint_dict[species][_] for _ in viewpoint_tag_list]
        )
        for species in species_tag_list
    ]

    viewpoint_order_list = ['left', 'frontleft', 'front', 'frontright', 'right', 'backright', 'back', 'backleft']
    pie_left_list = ['left', 'frontleft', 'front', 'frontright', 'right', 'backright', 'back', 'backleft']
    pie_right_list = ['right', 'backright', 'back', 'backleft', 'left', 'frontleft', 'front', 'frontright']
    viewpoint_tag_dict = {
        'zebra_grevys' : pie_right_list,
        'zebra_plains' : pie_left_list,
        'giraffe_masai' : pie_left_list,
    }
    pie_label_corrected_list = list(map(
        str,
        ['Correct', '+45', '+90', '+135', '+180', '+225', '+270', '+315']
    ))
    pie_values_corrected_list = [
        (
            species_nice_dict[species],
            [viewpoint_dict[species][_] for _ in viewpoint_tag_dict[species]]
        )
        for species in species_tag_list
    ]

    # Get number of annotations per name as a histogram for each species
    embedded = dict(globals(), **locals())
    return appf.template('view', 'viewpoints',
                         __wrapper_header__=False,
                         **embedded)


@register_route('/view/advanced/0/', methods=['GET'])
def view_advanced0(**kwargs):
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

    def filter_annots_general(aid_list):
        # Grevy's
        filter_kw = {
            'multiple': None,
            'minqual': 'good',
            'is_known': True,
            'min_pername': 1,
            'species': 'zebra_grevys',
            'view': ['right'],
        }
        aid_list1 = ibs.filter_annots_general(aid_list, filter_kw=filter_kw)

        # Plains
        filter_kw = {
            'multiple': None,
            'minqual': 'ok',
            'is_known': True,
            'min_pername': 1,
            'species': 'zebra_plains',
            'view': ['left'],
        }
        aid_list2 = ibs.filter_annots_general(aid_list, filter_kw=filter_kw)

        # Masai
        filter_kw = {
            'multiple': None,
            'minqual': 'ok',
            'is_known': True,
            'min_pername': 1,
            'species': 'giraffe_masai',
            'view': ['left'],
        }
        aid_list3 = ibs.filter_annots_general(aid_list, filter_kw=filter_kw)

        aid_list = list(set(aid_list1 + aid_list2 + aid_list3))

        # aid_list = ibs.filter_annots_general(aid_list, filter_kw=filter_kw)
        return aid_list

    ibs = current_app.ibs

    aid_list = ibs.get_valid_aids()
    aid_list = filter_annots_general(aid_list)
    aid_list = filter_annots_imageset(aid_list)
    gid_list = ibs.get_annot_gids(aid_list)
    unixtime_list = ibs.get_image_unixtime(gid_list)
    nid_list = ibs.get_annot_name_rowids(aid_list)
    date_list = _date_list(gid_list)

    # flagged_date_list = None
    flagged_date_list = ['2015/03/01', '2015/03/02', '2016/01/30', '2016/01/31']

    gid_list_unique = list(set(gid_list))
    date_list_unique = _date_list(gid_list_unique)
    date_taken_dict = {}
    for gid, date in zip(gid_list_unique, date_list_unique):
        if flagged_date_list is not None and date not in flagged_date_list:
            continue
        if date not in date_taken_dict:
            date_taken_dict[date] = [0, 0]
        date_taken_dict[date][1] += 1

    gid_list_all = ibs.get_valid_gids()
    gid_list_all = filter_images_imageset(gid_list_all)
    date_list_all = _date_list(gid_list_all)
    for gid, date in zip(gid_list_all, date_list_all):
        if flagged_date_list is not None and date not in flagged_date_list:
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
        if flagged_date_list is not None and date not in flagged_date_list:
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
    # contributor_list = ibs.get_valid_contributor_rowids()
    note_list = ibs.get_image_notes(gid_list)
    note_list = [
        ','.join(note.split(',')[:-1])
        for note in note_list
    ]
    contributor_list = set(note_list)
    # nid_list = ibs.get_valid_nids()
    aid_list_count = filter_annots_general(aid_list)
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
            if True or flagged_date_list is None:
                raise ValueError()
            index1 = bar_label_list.index(flagged_date_list[-3])
            index2 = bar_label_list.index(flagged_date_list[-2])
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

    gps_list_markers = [ gps for gps in gps_list_markers ]
    gps_list_markers_tuple_all = [ (gps, gid) for gps, gid in zip(gps_list_markers_all, gid_list)  ]
    gps_list_markers_all = [_[0] for _ in gps_list_markers_tuple_all]
    gid_list_markers_all = [_[1] for _ in gps_list_markers_tuple_all]
    gps_list_tracks = [
        [ gps for gps in gps_list_track ]
        for gps_list_track in gps_list_tracks
    ]

    VERSION = 2
    # Colors for GPS
    color_none = [0, "#777777"]
    color_day1 = [1, "#CA4141"]
    color_day2 = [2, "#428BCA"]
    color_resight = [3, "#9A41CA"]
    combined_list_markers_all = []
    dataset_color_dict = {}
    if VERSION == 1:
        dataset_color_label_list = ['Day 1 Only', 'Resightings', 'Day 2 Only']
    else:
        dataset_color_label_list = ['Day 1 Only', 'Resightings', 'Day 2 Only', 'Unused']

    for gid, gps in zip(gid_list_markers_all, gps_list_markers_all):
        image_note = ibs.get_image_notes(gid)
        current_date = _date_list([gid])[0]
        color = color_none
        if current_date not in flagged_date_list:
            color = color_none
        elif gps == (-1, -1):
            color = color_none
        else:
            aid_list_ = ibs.get_image_aids(gid)
            nid_list_ = ibs.get_annot_nids(aid_list_)
            nid_list_ = [nid for nid in nid_list_ if nid > 0]
            if len(nid_list_) == 0:
                color = color_none
            else:
                aid_list_ = ut.flatten(ibs.get_name_aids(nid_list_))
                gid_list_ = list(set(ibs.get_annot_gids(aid_list_)))
                date_list = set(_date_list(gid_list_))
                if current_date == '2015/03/01':
                    if '2015/03/02' in date_list:
                        color = color_resight
                    else:
                        color = color_day1
                elif current_date == '2015/03/02':
                    if '2015/03/01' in date_list:
                        color = color_resight
                    else:
                        color = color_day2
                elif current_date == '2016/01/30':
                    if '2016/01/31' in date_list:
                        color = color_resight
                    else:
                        color = color_day1
                elif current_date == '2016/01/31':
                    if '2016/01/30' in date_list:
                        color = color_resight
                    else:
                        color = color_day2

        color_id = color[0]
        if VERSION == 1 and color_id == 0:
            continue
        if VERSION == 1:
            color_id -= 1

        combined_list_markers_all.append(tuple(color + [gps]))
        dataset_tag = 'GGR' if 'GGR' in image_note else 'GZGC'

        if dataset_tag not in dataset_color_dict:
            if VERSION == 1:
                dataset_color_dict[dataset_tag] = [0, 0, 0]
            else:
                dataset_color_dict[dataset_tag] = [0, 0, 0, 0]
        dataset_color_dict[dataset_tag][color_id] += 1

    for dataset_tag in dataset_color_dict:
        temp = dataset_color_dict[dataset_tag]
        temp[-1], temp[-2] = temp[-2], temp[-1]
        if len(temp) == 4:
            temp = temp[1:] + temp[:1]
        dataset_color_dict[dataset_tag] = temp

    combined_list_markers_all = sorted(combined_list_markers_all)
    marker = None
    for index, combined in enumerate(combined_list_markers_all):
        if combined[0] == 0:
            continue
        marker = index
        break

    # Shuffle day 1, day 2, resights randomly
    sublist = combined_list_markers_all[marker:]
    random.shuffle(sublist)
    combined_list_markers_all[marker:] = sublist

    color_list_markers_all = [_[1] for _ in combined_list_markers_all]
    gps_list_markers_all = [_[2] for _ in combined_list_markers_all]
    assert len(color_list_markers_all) == len(gps_list_markers_all)

    JITTER_GPS = True
    JITTER_AMOUNT = 0.0001
    if JITTER_GPS:
        gps_list_markers_all = [
            gps if color_ == color_none[1] else
            [
                gps[0] + random.uniform(-JITTER_AMOUNT, JITTER_AMOUNT),
                gps[1] + random.uniform(-JITTER_AMOUNT, JITTER_AMOUNT),
            ]
            for gps, color_ in zip(gps_list_markers_all, color_list_markers_all)
        ]

    valid_aids = ibs.get_valid_aids()
    valid_aids = filter_annots_imageset(valid_aids)
    used_gids = list(set( ibs.get_annot_gids(valid_aids) ))
    # used_contributor_tags = list(set( ibs.get_image_contributor_tag(used_gids) ))
    note_list = ibs.get_image_notes(used_gids)
    note_list = [
        ','.join(note.split(',')[:-1])
        for note in note_list
    ]
    used_contributor_tags = set(note_list)

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

    return appf.template('view', 'advanced0',
                         line_index_list=index_list,
                         line_label_list=label_list,
                         line_value_list=value_list,
                         prediction_list=prediction_list,
                         pl_index=pl_index,
                         pl_error=pl_error,
                         gps_list_markers=gps_list_markers,
                         gps_list_markers_all=gps_list_markers_all,
                         color_list_markers_all=color_list_markers_all,
                         gps_list_tracks=gps_list_tracks,
                         dataset_color_dict=dataset_color_dict,
                         dataset_color_label_list=dataset_color_label_list,
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
                         contributor_list=contributor_list,
                         contributor_list_str=','.join(map(str, contributor_list)),
                         num_contribs=len(contributor_list),
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
                         used_contribs=used_contributor_tags,
                         num_used_contribs=len(used_contributor_tags),
                         __wrapper_header__=False)


@register_route('/view/advanced/1/', methods=['GET'])
def view_advanced1(**kwargs):
    ibs = current_app.ibs

    species_tag_list = ['zebra_grevys', 'zebra_plains', 'giraffe_masai']
    species_rowid_list = ibs.get_species_rowids_from_text(species_tag_list)
    species_set = set(species_tag_list)
    species_nice_dict = {
        species_tag : ibs.get_species_nice(species_rowid)
        for species_tag, species_rowid in zip(species_tag_list, species_rowid_list)
    }

    aid_list = ibs.get_valid_aids()
    species_list = ibs.get_annot_species_texts(aid_list)
    viewpoint_list = ibs.get_annot_viewpoints(aid_list)
    viewpoint_dict = {}

    for species, viewpoint in zip(species_list, viewpoint_list):
        if species in species_set:
            if species not in viewpoint_dict:
                viewpoint_dict[species] = {}
            if viewpoint not in viewpoint_dict[species]:
                viewpoint_dict[species][viewpoint] = 0
            viewpoint_dict[species][viewpoint] += 1

    viewpoint_tag_list = const.VIEWTEXT_TO_VIEWPOINT_RADIANS.keys()
    pie_label_list = [ str(const.VIEWPOINTALIAS_NICE[_]) for _ in viewpoint_tag_list ]
    pie_values_list = [
        (
            species_nice_dict[species],
            [viewpoint_dict[species][_] for _ in viewpoint_tag_list]
        )
        for species in species_tag_list
    ]

    pie_left_list = ['left', 'frontleft', 'front', 'frontright', 'right', 'backright', 'back', 'backleft']
    pie_right_list = ['right', 'backright', 'back', 'backleft', 'left', 'frontleft', 'front', 'frontright']
    viewpoint_tag_dict = {
        'zebra_grevys' : pie_right_list,
        'zebra_plains' : pie_left_list,
        'giraffe_masai' : pie_left_list,
    }
    pie_label_corrected_list = list(map(
        str,
        ['Correct', '+45', '+90', '+135', '+180', '+225', '+270', '+315']
    ))
    pie_values_corrected_list = [
        (
            species_nice_dict[species],
            [viewpoint_dict[species][_] for _ in viewpoint_tag_dict[species]]
        )
        for species in species_tag_list
    ]

    gid_list = ibs.get_valid_gids()
    note_list = ibs.get_image_notes(gid_list)
    aids_list = ibs.get_image_aids(gid_list)
    viewpoints_list = ut.unflat_map(ibs.get_annot_viewpoints, aids_list)
    dataset_tag_list = ['GGR', 'GZGC']
    pie_label_images_list = ['Correct Viewpoint', '+/- 45 Viewpoint', 'Unused']
    pie_values_images_dict = {
        dataset_tag : [0, 0, 0]
        for dataset_tag in dataset_tag_list
    }
    allowed_viewpoint_dict = {
        'GGR': ['right', 'frontright', 'backright'],
        'GZGC': ['left', 'frontleft', 'backleft'],
    }
    for note, viewpoint_list in zip(note_list, viewpoints_list):
        dataset_tag = 'GGR' if 'GGR' in note else 'GZGC'
        allowed_viewpoint_list = allowed_viewpoint_dict[dataset_tag]

        found = False
        for index, allowed_viewpoint in enumerate(allowed_viewpoint_list):
            if allowed_viewpoint in viewpoint_list:
                found = True
                if index == 0:
                    pie_values_images_dict[dataset_tag][0] += 1
                else:
                    pie_values_images_dict[dataset_tag][1] += 1
                break
        if not found:
            pie_values_images_dict[dataset_tag][2] += 1

    pie_values_images_list = [
        (_, pie_values_images_dict[_])
        for _ in dataset_tag_list
    ]

    nid_list = ibs.get_valid_nids()
    aids_list = ibs.get_name_aids(nid_list)
    species_list = map(list, map(set, ut.unflat_map(ibs.get_annot_species_texts, aids_list)))
    species_list = [ None if len(_) != 1 else _[0] for _ in species_list ]

    num_bins = 15
    histogram_dict = {}
    for nid, aids, species in zip(nid_list, aids_list, species_list):
        if species not in histogram_dict:
            histogram_dict[species] = {}
        count = len(aids)
        count = min(count, num_bins)
        if count not in histogram_dict[species]:
            histogram_dict[species][count] = 0
        histogram_dict[species][count] += 1

    histogram_bins = list(range(1, num_bins + 1))
    bar_label_list = [
        '%s+' % (bin_, ) if bin_ == histogram_bins[-1] else '%s' % (bin_, )
        for bin_ in histogram_bins
    ]
    bar_values_dict = {}
    for species in histogram_dict:
        bar_values_dict[species] = []
        for bin_ in histogram_bins:
            value = histogram_dict[species].get(bin_, 0)
            if species == 'zebra_plains':
                value2 = histogram_dict['giraffe_masai'].get(bin_, 0)
                value += value2
            bar_values_dict[species].append(value)

    # Get number of annotations per name as a histogram for each species
    embedded = dict(globals(), **locals())
    return appf.template('view', 'advanced1',
                         __wrapper_header__=False,
                         **embedded)


@register_route('/view/advanced/2/', methods=['GET'])
def view_advanced2(**kwargs):
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

    def filter_annots_general(aid_list):
        # Grevy's
        filter_kw = {
            'multiple': None,
            'minqual': 'good',
            'is_known': True,
            'min_pername': 1,
            'species': 'zebra_grevys',
            'view': ['right'],
        }
        aid_list1 = ibs.filter_annots_general(aid_list, filter_kw=filter_kw)
        # aid_list1 = []

        # Plains
        filter_kw = {
            'multiple': None,
            'minqual': 'ok',
            'is_known': True,
            'min_pername': 1,
            'species': 'zebra_plains',
            'view': ['left'],
        }
        aid_list2 = ibs.filter_annots_general(aid_list, filter_kw=filter_kw)
        aid_list2 = []

        # Masai
        filter_kw = {
            'multiple': None,
            'minqual': 'ok',
            'is_known': True,
            'min_pername': 1,
            'species': 'giraffe_masai',
            'view': ['left'],
        }
        aid_list3 = ibs.filter_annots_general(aid_list, filter_kw=filter_kw)
        aid_list3 = []

        aid_list = aid_list1 + aid_list2 + aid_list3

        # aid_list = ibs.filter_annots_general(aid_list, filter_kw=filter_kw)
        return aid_list

    ibs = current_app.ibs

    aid_list = ibs.get_valid_aids()
    aid_list = filter_annots_general(aid_list)
    aid_list = filter_annots_imageset(aid_list)
    species_list = ibs.get_annot_species_texts(aid_list)
    gid_list = ibs.get_annot_gids(aid_list)
    unixtime_list = ibs.get_image_unixtime(gid_list)
    nid_list = ibs.get_annot_name_rowids(aid_list)
    date_list = _date_list(gid_list)

    flagged_date_list = ['2015/03/01', '2015/03/02', '2016/01/30', '2016/01/31']

    index = 0
    value = 0
    line_index_list = []
    line_label_list = []
    line_value_list = []
    seen_set = set()
    last_date = None
    for unixtime, aid, nid, date, species in sorted(list(zip(unixtime_list, aid_list, nid_list, date_list, species_list))):
        # if flagged_date_list is not None and date not in flagged_date_list:
        #     continue

        index += 1
        line_index_list.append(index)

        # Add to counters
        if nid not in seen_set:
            seen_set.add(nid)
            value += 1

        # Add to register
        line_value_list.append(value)

        # Reset step (per day)
        if index % 1000 == 0:
            line_label_list.append(index)
        else:
            line_label_list.append('')
        # if date != last_date and date != 'UNKNOWN':
        #     last_date = date
        #     # line_label_list.append(date)
        #     line_label_list.append('')
        # else:
        #     line_label_list.append('')

    # Get number of annotations per name as a histogram for each species
    embedded = dict(globals(), **locals())
    return appf.template('view', 'advanced2',
                         __wrapper_header__=False,
                         **embedded)


@register_route('/view/advanced/3/', methods=['GET'])
def view_advanced3(**kwargs):

    ibs = current_app.ibs

    gid_list = ibs.get_valid_gids()
    # gid_list = gid_list[:100] + gid_list[-100:]
    note_list = ibs.get_image_notes(gid_list)

    contrib_dict = {}
    skipped = 0
    for gid, note in zip(gid_list, note_list):
        note = note.strip()
        if len(note) == 0:
            skipped += 1
            continue

        dataset_tag = 'GGR' if 'GGR' in note else 'GZGC'
        if dataset_tag not in contrib_dict:
            contrib_dict[dataset_tag] = {}

        if dataset_tag == 'GGR':
            note_ = note.strip().split(',')
            car, letter = note_[1:3]
        else:
            note_ = note.strip().split(',')
            car, letter = note_[0:2]
            car = car.split(' ')[-1].strip('\'')
            letter = letter.split(' ')[-1].strip('\'')

        if car not in contrib_dict[dataset_tag]:
            contrib_dict[dataset_tag][car] = {}
        if letter not in contrib_dict[dataset_tag][car]:
            contrib_dict[dataset_tag][car][letter] = 0

        contrib_dict[dataset_tag][car][letter] += 1

    max_size = 0
    for dataset_tag in contrib_dict:
        temp_list = []
        for car in contrib_dict[dataset_tag]:
            letter_dict = contrib_dict[dataset_tag][car]
            combined_list = list([(_[1], _[0]) for _ in letter_dict.items()])
            combined_list = sorted(combined_list, reverse=True)
            letter_list = [_[1] for _ in combined_list]
            total = sum(letter_dict.values())
            temp_list.append((total, car, letter_list))
        temp_list = sorted(temp_list, reverse=True)
        max_size = max(max_size, len(temp_list))
        contrib_dict[dataset_tag]['__MANIFEST__'] = temp_list

    max_show = 30
    # bar_label_list = [''] * max_show
    bar_label_list = list(range(1, max_show + 1))
    bar_values_dict = {}
    for dataset_tag in contrib_dict:
        values = [_[0] for _ in contrib_dict[dataset_tag]['__MANIFEST__']]
        padding = max_size - len(values)
        values = values + [0] * padding
        values = values[:max_show]
        bar_values_dict[dataset_tag] = values

    for dataset_tag in contrib_dict:
        print(dataset_tag)
        total_cars = 0
        total_letters = 0
        total_images = 0
        for car in contrib_dict[dataset_tag]:
            if car == '__MANIFEST__':
                continue
            total_cars += 1
            for letter in contrib_dict[dataset_tag][car]:
                total_letters += 1
                total_images += contrib_dict[dataset_tag][car][letter]
        print(total_cars)
        print(total_letters)
        print(total_images)

    print(skipped)

    # Get number of annotations per name as a histogram for each species
    embedded = dict(globals(), **locals())
    return appf.template('view', 'advanced3',
                         __wrapper_header__=False,
                         **embedded)


@register_route('/view/advanced/4/', methods=['GET'])
def view_advanced4(**kwargs):

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

    def filter_species_of_interest(gid_list):
        wanted_set = set(['zebra_plains', 'zebra_grevys', 'giraffe_masai'])
        aids_list = ibs.get_image_aids(gid_list)
        speciess_list = ut.unflat_map(ibs.get_annot_species_texts, aids_list)
        speciess_set = map(set, speciess_list)
        gid_list_filtered = []
        for gid, species_set in zip(gid_list, speciess_set):
            intersect_list = list(wanted_set & species_set)
            if len(intersect_list) > 0:
                gid_list_filtered.append(gid)
        return gid_list_filtered

    def filter_viewpoints_of_interest(gid_list, allowed_viewpoint_list):
        aids_list = ibs.get_image_aids(gid_list)
        wanted_set = set(allowed_viewpoint_list)
        viewpoints_list = ut.unflat_map(ibs.get_annot_viewpoints, aids_list)
        viewpoints_list = map(set, viewpoints_list)
        gid_list_filtered = []
        for gid, viewpoint_set in zip(gid_list, viewpoints_list):
            intersect_list = list(wanted_set & viewpoint_set)
            if len(intersect_list) > 0:
                gid_list_filtered.append(gid)
        return gid_list_filtered

    def filter_bad_metadata(gid_list):
        wanted_set = set(['2015/03/01', '2015/03/02', '2016/01/30', '2016/01/31'])
        date_list = _date_list(gid_list)
        gps_list = ibs.get_image_gps(gid_list)
        gid_list_filtered = []
        for gid, date, gps in zip(gid_list, date_list, gps_list):
            if date in wanted_set and gps != (-1.0, -1.0):
                gid_list_filtered.append(gid)
        return gid_list_filtered

    def filter_bad_quality(gid_list, allowed_quality_list):
        aids_list = ibs.get_image_aids(gid_list)
        wanted_set = set(allowed_quality_list)
        qualities_list = ut.unflat_map(ibs.get_annot_quality_texts, aids_list)
        qualities_list = map(set, qualities_list)
        gid_list_filtered = []
        for gid, quality_list in zip(gid_list, qualities_list):
            intersect_list = list(wanted_set & quality_list)
            if len(intersect_list) > 0:
                gid_list_filtered.append(gid)
        return gid_list_filtered

    # def filter_singletons(gid_list):
    #     aids_list = ibs.get_image_aids(gid_list)
    #     nids_list = ut.unflat_map(ibs.get_annot_nids, aids_list)
    #     gid_list_filtered = []
    #     for gid, nid_list in zip(gid_list, nids_list):
    #         print(gid)
    #         print(nid_list)
    #         aids_list_ = ibs.get_name_aids(nid_list)
    #         print(aids_list_)
    #         single = True
    #         for nid, aid_list in zip(nid_list, aids_list_):
    #             if nid == const.UNKNOWN_NAME_ROWID or nid < 0:
    #                 continue
    #             if len(aid_list) > 1:
    #                 single = False
    #                 break
    #         print(single)
    #         if single:
    #             gid_list_filtered.append(gid)
    #     return gid_list_filtered

    ibs = current_app.ibs

    gid_list = ibs.get_valid_gids()
    note_list = ibs.get_image_notes(gid_list)

    dataset_dict = {}
    skipped = 0
    for gid, note in zip(gid_list, note_list):
        note = note.strip()

        dataset_tag = 'GGR' if 'GGR' in note else 'GZGC'
        if dataset_tag not in dataset_dict:
            dataset_dict[dataset_tag] = []

        dataset_dict[dataset_tag].append(gid)

    num_all_gzgc = len(dataset_dict['GZGC'])
    num_all_ggr = len(dataset_dict['GGR'])

    print('all', num_all_gzgc)
    print('all', num_all_ggr)

    dataset_dict['GZGC'] = filter_species_of_interest(dataset_dict['GZGC'])
    dataset_dict['GGR'] = filter_species_of_interest(dataset_dict['GGR'])

    num_species_gzgc = len(dataset_dict['GZGC'])
    num_species_ggr = len(dataset_dict['GGR'])

    print('species', num_species_gzgc)
    print('species', num_species_ggr)

    allowed_viewpoint_dict = {
        'GGR': ['right', 'frontright', 'backright'],
        'GZGC': ['left', 'frontleft', 'backleft'],
    }
    dataset_dict['GZGC'] = filter_viewpoints_of_interest(dataset_dict['GZGC'], allowed_viewpoint_dict['GZGC'])
    dataset_dict['GGR'] = filter_viewpoints_of_interest(dataset_dict['GGR'], allowed_viewpoint_dict['GGR'])

    num_viewpoint_gzgc = len(dataset_dict['GZGC'])
    num_viewpoint_ggr = len(dataset_dict['GGR'])

    print('viewpoint', num_viewpoint_gzgc)
    print('viewpoint', num_viewpoint_ggr)

    allowed_quality_dict = {
        'GGR': ['good', 'perfect'],
        'GZGC': ['ok', 'good', 'perfect'],
    }
    dataset_dict['GZGC'] = filter_bad_quality(dataset_dict['GZGC'], allowed_quality_dict['GZGC'])
    dataset_dict['GGR'] = filter_bad_quality(dataset_dict['GGR'], allowed_quality_dict['GZGC'])

    num_quality_gzgc = len(dataset_dict['GZGC'])
    num_quality_ggr = len(dataset_dict['GGR'])

    print('quality', num_quality_gzgc)
    print('quality', num_quality_ggr)

    dataset_dict['GZGC'] = filter_bad_metadata(dataset_dict['GZGC'])
    dataset_dict['GGR'] = filter_bad_metadata(dataset_dict['GGR'])

    num_metadata_gzgc = len(dataset_dict['GZGC'])
    num_metadata_ggr = len(dataset_dict['GGR'])

    print('metadata', num_metadata_gzgc)
    print('metadata', num_metadata_ggr)

    # dataset_dict['GZGC'] = filter_singletons(dataset_dict['GZGC'])
    # dataset_dict['GGR'] = filter_singletons(dataset_dict['GGR'])

    # num_named_gzgc = len(dataset_dict['GZGC'])
    # num_named_ggr = len(dataset_dict['GGR'])

    # print('named', num_named_gzgc)
    # print('named', num_named_ggr)

    stage_list_gzgc = [
        num_all_gzgc,
        num_species_gzgc,
        num_viewpoint_gzgc,
        num_quality_gzgc,
        num_metadata_gzgc,
        # num_named_gzgc,
        0,
    ]

    stage_list_ggr = [
        num_all_ggr,
        num_species_ggr,
        num_viewpoint_ggr,
        num_quality_ggr,
        num_metadata_ggr,
        # num_named_ggr,
        0,
    ]

    diff_list_gzgc = [
        stage_list_gzgc[i] - stage_list_gzgc[i + 1]
        for i in range(len(stage_list_gzgc) - 1)
    ]

    diff_list_ggr = [
        stage_list_ggr[i] - stage_list_ggr[i + 1]
        for i in range(len(stage_list_ggr) - 1)
    ]

    bar_value_list = map(list, zip(diff_list_gzgc, diff_list_ggr))

    # Get number of annotations per name as a histogram for each species
    embedded = dict(globals(), **locals())
    return appf.template('view', 'advanced4',
                         __wrapper_header__=False,
                         **embedded)


@register_route('/view/imagesets/', methods=['GET'])
def view_imagesets(**kwargs):
    ibs = current_app.ibs
    filtered = True
    ibs.update_all_image_special_imageset()
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
    is_normal_list = ut.not_list(ibs.is_special_imageset(imgsetid_list))
    imageset_list = zip(
        is_normal_list,
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
    imageset_list.sort(key=lambda t: t[0])
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
def view_images(**kwargs):
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
def view_annotations(**kwargs):
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
        ibs.get_annot_viewpoints(aid_list),
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
def view_names(**kwargs):
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
        ibs.get_annot_viewpoints(aid_list_),
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


@register_route('/action/', methods=['GET'])
def action(**kwargs):
    ibs = current_app.ibs

    imgsetid = request.args.get('imgsetid', '')
    imgsetid = None if imgsetid == 'None' or imgsetid == '' else int(imgsetid)

    job_id_list = ibs.get_job_id_list()
    job_action_list = [None] * len(job_id_list)
    job_status_list = [
        ibs.get_job_status(job_id)['jobstatus']
        for job_id in job_id_list
    ]
    action_list = zip(job_id_list, job_action_list, job_status_list)

    return appf.template('action', None,
                         imgsetid=imgsetid,
                         action_list=action_list,
                         num_actions=len(action_list))


@register_route('/action/detect/', methods=['GET'])
def action_detect(**kwargs):
    ibs = current_app.ibs

    gid_list = ibs.get_valid_gids()
    image_uuid_list = ibs.get_image_uuids(gid_list)
    ibs.start_detect_image(image_uuid_list)

    return redirect(url_for('action'))


@register_route('/action/identify/', methods=['GET'])
def action_identification(**kwargs):
    ibs = current_app.ibs

    ibs.start_web_query_all()

    return redirect(url_for('action'))


@register_route('/turk/', methods=['GET'])
def turk(imgsetid=None):
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
def turk_detection(gid=None, refer_aid=None, imgsetid=None, previous=None, config=None, **kwargs):

    ibs = current_app.ibs

    if config is None:
        config = {
            'autointerest': True,
            'metadata': False,
            'parts': False,
        }

    imgsetid = None if imgsetid == '' or imgsetid == 'None' else imgsetid
    gid_list = ibs.get_valid_gids(imgsetid=imgsetid)
    reviewed_list = appf.imageset_image_processed(ibs, gid_list)
    try:
        progress = '%0.2f' % (100.0 * reviewed_list.count(True) / len(gid_list), )
    except ZeroDivisionError:
        progress = '100.0'

    imagesettext = None if imgsetid is None else ibs.get_imageset_text(imgsetid)
    if gid is None:
        gid_list_ = ut.filterfalse_items(gid_list, reviewed_list)
        if len(gid_list_) == 0:
            gid = None
        else:
            # gid = gid_list_[0]
            gid = random.choice(gid_list_)
    finished = gid is None
    review = 'review' in request.args.keys()
    display_instructions = request.cookies.get('ia-detection_instructions_seen', 1) == 0
    display_species_examples = False  # request.cookies.get('ia-detection_example_species_seen', 0) == 0
    if not finished:
        gpath = ibs.get_image_thumbpath(gid, ensure_paths=True, draw_annots=False)
        imgdata = ibs.get_image_imgdata(gid)
        image_src = appf.embed_image_html(imgdata)
        width, height = ibs.get_image_sizes(gid)

        # Get annotations
        aid_list = ibs.get_image_aids(gid)
        annot_bbox_list = ibs.get_annot_bboxes(aid_list)
        annot_theta_list = ibs.get_annot_thetas(aid_list)
        species_list = ibs.get_annot_species_texts(aid_list)
        viewpoint_list = ibs.get_annot_viewpoints(aid_list)
        quality_list = ibs.get_annot_qualities(aid_list)
        multiple_list = ibs.get_annot_multiple(aid_list)
        interest_list = ibs.get_annot_interest(aid_list)
        # Get annotation bounding boxes
        mapping_dict = {}
        annotation_list = []
        zipped = zip(aid_list, annot_bbox_list, annot_theta_list, species_list, viewpoint_list, quality_list, multiple_list, interest_list)
        for aid, annot_bbox, annot_theta, species, viewpoint, quality, multiple, interest in zipped:
            if quality in [-1, None]:
                quality = 0
            elif quality <= 2:
                quality = 1
            elif quality > 2:
                quality = 2

            viewpoint1, viewpoint2, viewpoint3 = appf.convert_viewpoint_to_tuple(viewpoint)

            temp = {}
            temp['id']         = aid
            temp['left']       = 100.0 * (annot_bbox[0] / width)
            temp['top']        = 100.0 * (annot_bbox[1] / height)
            temp['width']      = 100.0 * (annot_bbox[2] / width)
            temp['height']     = 100.0 * (annot_bbox[3] / height)
            temp['theta']      = float(annot_theta)
            temp['viewpoint1'] = viewpoint1
            temp['viewpoint2'] = viewpoint2
            temp['viewpoint3'] = viewpoint3
            temp['quality']    = quality
            temp['multiple']   = 'true' if multiple == 1 else 'false'
            temp['interest']   = 'true' if interest == 1 else 'false'
            temp['species']    = species

            mapping_dict[aid] = len(annotation_list)
            annotation_list.append(temp)

        # Get parts
        part_rowid_list = ut.flatten(ibs.get_annot_part_rowids(aid_list))
        part_aid_list = ibs.get_part_aids(part_rowid_list)
        part_bbox_list = ibs.get_part_bboxes(part_rowid_list)
        part_theta_list = ibs.get_part_thetas(part_rowid_list)
        part_viewpoint_list = ibs.get_part_viewpoints(part_rowid_list)
        part_quality_list = ibs.get_part_qualities(part_rowid_list)
        part_type_list = ibs.get_part_types(part_rowid_list)
        # Get annotation bounding boxes

        part_list = []
        zipped = zip(part_rowid_list, part_aid_list, part_bbox_list, part_theta_list, part_viewpoint_list, part_quality_list, part_type_list)
        for part_rowid, part_aid, part_bbox, part_theta, part_viewpoint, part_quality, part_type in zipped:
            if part_quality in [-1, None]:
                part_quality = 0
            elif part_quality <= 2:
                part_quality = 1
            elif part_quality > 2:
                part_quality = 2

            viewpoint1, viewpoint2, viewpoint3 = appf.convert_viewpoint_to_tuple(part_viewpoint)

            temp = {}
            temp['id']         = part_rowid
            temp['parent']     = mapping_dict[part_aid]
            temp['left']       = 100.0 * (part_bbox[0] / width)
            temp['top']        = 100.0 * (part_bbox[1] / height)
            temp['width']      = 100.0 * (part_bbox[2] / width)
            temp['height']     = 100.0 * (part_bbox[3] / height)
            temp['theta']      = float(part_theta)
            temp['viewpoint1'] = viewpoint1
            temp['quality']    = part_quality
            temp['type']       = part_type
            part_list.append(temp)

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
        part_list = []

    species_rowids = ibs._get_all_species_rowids()
    species_nice_list = ibs.get_species_nice(species_rowids)

    combined_list = sorted(zip(species_nice_list, species_rowids))
    species_nice_list = [ combined[0] for combined in combined_list ]
    species_rowids = [ combined[1] for combined in combined_list ]

    species_text_list = ibs.get_species_texts(species_rowids)
    species_list = zip(species_nice_list, species_text_list)
    species_list = [ ('Unspecified', const.UNKNOWN) ] + species_list

    # Collect mapping of species to parts
    aid_list = ibs.get_valid_aids()
    part_species_rowid_list = ibs.get_annot_species_rowids(aid_list)
    part_species_text_list = ibs.get_species_texts(part_species_rowid_list)
    part_rowids_list = ibs.get_annot_part_rowids(aid_list)
    part_types_list = map(ibs.get_part_types, part_rowids_list)

    zipped = zip(part_species_text_list, part_types_list)
    species_part_dict = {
        const.UNKNOWN: set([])
    }
    for part_species_text, part_type_list in zipped:
        if part_species_text not in species_part_dict:
            species_part_dict[part_species_text] = set([const.UNKNOWN])
        for part_type in part_type_list:
            species_part_dict[part_species_text].add(part_type)
            species_part_dict[const.UNKNOWN].add(part_type)
    # Add any images that did not get added because they aren't assigned any annotations
    for species_text in species_text_list:
        if species_text not in species_part_dict:
            species_part_dict[species_text] = set([const.UNKNOWN])
    for key in species_part_dict:
        species_part_dict[key] = sorted(list(species_part_dict[key]))
    species_part_dict_json = json.dumps(species_part_dict)

    orientation_flag = '0'
    if species is not None and 'zebra' in species:
        orientation_flag = '1'

    settings_key_list = [
        ('ia-detection-setting-orientation', orientation_flag),
        ('ia-detection-setting-parts-assignments', '1'),
        ('ia-detection-setting-toggle-annotations', '1'),
        ('ia-detection-setting-toggle-parts', '0'),
        ('ia-detection-setting-parts-show', '0'),
        ('ia-detection-setting-parts-hide', '0'),
    ]

    settings = {
        settings_key: request.cookies.get(settings_key, settings_default) == '1'
        for (settings_key, settings_default) in settings_key_list
    }

    callback_url = '%s?imgsetid=%s' % (url_for('submit_detection'), imgsetid, )
    return appf.template('turk', 'detection',
                         imgsetid=imgsetid,
                         gid=gid,
                         config=config,
                         refer_aid=refer_aid,
                         species=species,
                         image_path=gpath,
                         image_src=image_src,
                         previous=previous,
                         imagesettext=imagesettext,
                         progress=progress,
                         finished=finished,
                         species_list=species_list,
                         species_part_dict_json=species_part_dict_json,
                         annotation_list=annotation_list,
                         part_list=part_list,
                         display_instructions=display_instructions,
                         display_species_examples=display_species_examples,
                         settings=settings,
                         callback_url=callback_url,
                         callback_method='POST',
                         EMBEDDED_CSS=None,
                         EMBEDDED_JAVASCRIPT=None,
                         review=review)


@register_route('/turk/detection/dynamic/', methods=['GET'])
def turk_detection_dynamic(**kwargs):
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
def turk_annotation(**kwargs):
    """
    CommandLine:
        python -m ibeis.web.app --exec-turk_annotation --db PZ_Master1

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
        viewpoint_text = ibs.get_annot_viewpoints(aid)
        viewpoint_value = appf.VIEWPOINT_MAPPING_INVERT.get(viewpoint_text, None)
        quality_value = ibs.get_annot_qualities(aid)
        if quality_value in [-1, None]:
            quality_value = -1
        elif quality_value > 2:
            quality_value = 1
        elif quality_value <= 2:
            quality_value = 0
        multiple_value = ibs.get_annot_multiple(aid) == 1
    else:
        try:
            ibs.update_special_imagesets()
            ibs.notify_observers()
        except:
            pass
        gid       = None
        gpath     = None
        image_src = None
        species   = None
        viewpoint_value = -1
        quality_value = -1
        multiple_value = False

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
def turk_annotation_dynamic(**kwargs):
    ibs = current_app.ibs
    aid = request.args.get('aid', None)
    imgsetid = request.args.get('imgsetid', None)

    review = 'review' in request.args.keys()
    gid       = ibs.get_annot_gids(aid)
    gpath     = ibs.get_annot_chip_fpath(aid)
    image     = vt.imread(gpath)
    image_src = appf.embed_image_html(image)
    species   = ibs.get_annot_species_texts(aid)
    viewpoint_text = ibs.get_annot_viewpoints(aid)
    viewpoint_value = appf.VIEWPOINT_MAPPING_INVERT.get(viewpoint_text, None)
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
def turk_viewpoint(**kwargs):
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

    viewpoint_text = ibs.get_annot_viewpoints(aid)
    value = appf.VIEWPOINT_MAPPING_INVERT.get(viewpoint_text, None)
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


def commit_current_query_object_names(query_object, ibs):
    r"""
    Args:
        query_object (ibeis.AnnotInference):
        ibs (ibeis.IBEISController):  image analysis api
    """
    # Ensure connected components are used to relabel names
    query_object.relabel_using_reviews()
    # Transfers any remaining internal feedback into staging
    # TODO:  uncomment once buffer is dead
    # query_object.write_ibeis_staging_feedback()
    # Commit a delta of the current annotmatch
    query_object.write_ibeis_annotmatch_feedback()
    query_object.write_ibeis_name_assignment()


def precompute_current_review_match_images(ibs, query_object,
                                           global_feedback_limit=GLOBAL_FEEDBACK_LIMIT,
                                           view_orientation='vertical'):
    from ibeis.web import apis_query

    review_aid1_list, review_aid2_list = query_object.get_filtered_edges(GLOBAL_FEEDBACK_CONFIG_DICT)
    qreq_ = query_object.qreq_

    assert len(review_aid1_list) == len(review_aid2_list), 'not aligned'

    # Precompute
    zipped = zip(review_aid1_list, review_aid2_list)
    prog = ut.ProgIter(enumerate(zipped), length=len(review_aid2_list),
                       label='Rending images')
    for index, (aid1, aid2) in prog:
        if index > global_feedback_limit * 2:
            break
        cm, aid1, aid2 = query_object.lookup_cm(aid1, aid2)
        try:
            apis_query.ensure_review_image(ibs, aid2, cm, qreq_,
                                           view_orientation=view_orientation)
        except KeyError as ex:
            ut.printex(ex, 'Failed to make review image. falling back',
                       tb=True, keys=['cm.qaid', 'aid2'], iswarning=True)
            apis_query.ensure_review_image(ibs, aid2, cm, qreq_,
                                           view_orientation=view_orientation,
                                           draw_matches=False)


@register_ibs_method
def load_identification_query_object_worker(ibs, **kwargs):
    _init_identification_query_object(ibs, **kwargs)
    return 'precomputed'


def _init_identification_query_object(ibs, debug_ignore_name_gt=False,
                                      global_feedback_limit=GLOBAL_FEEDBACK_LIMIT,
                                      **kwargs):
    """
    CommandLine:
        python -m ibeis.web.routes _init_identification_query_object

    Ignore:
        # mount lev to the home drive
        sshfs -o idmap=user lev:/ ~/lev

        # unmount
        fusermount -u ~/lev

        # Url to debug
        http://128.213.17.12:5000/turk/identification/?aid1=6619&aid2=7094

        import ibeis
        ibs = ibeis.opendb(ut.truepath('~/lev/media/hdd/work/EWT_Cheetahs'))
        aid_list = ibs.filter_annots_general(view=['right', 'frontright', 'backright'])
        infr = ibeis.AnnotInference(ibs, aid_list, autoinit=True)
        infr.reset_feedback('staging')
        infr.apply_feedback_edges()
        infr.exec_matching()
        infr.apply_match_edges()
        infr.apply_match_scores()

    Example:
        >>> # SLOW_DOCTEST
        >>> from ibeis.web.routes import *  # NOQA
        >>> from ibeis.web.routes import _init_identification_query_object
        >>> import ibeis
        >>> ibs = ibeis.opendb('PZ_MTEST')
        >>> ut.exec_funckw(_init_identification_query_object, locals())
        >>> kwargs = {}
        >>> _init_identification_query_object(ibs)

    """
    from ibeis.algo.graph import graph_iden

    if ibs.dbname == 'EWT_Cheetahs':
        aid_list = ibs.filter_annots_general(view=['right', 'frontright', 'backright'])
    else:
        aid_list = ibs.get_valid_aids(is_exemplar=True)

    nids = [-aid for aid in aid_list] if debug_ignore_name_gt else None

    # Initailize a graph with no edges.
    query_object = graph_iden.AnnotInference(ibs, aid_list, nids=nids,
                                             autoinit=True)
    # Load feedback from the staging database (does not change graph state)
    query_object.reset_feedback('staging')
    # Rectify inconsistent feedback and create decision edges
    query_object.apply_feedback_edges()
    # Exec matching (adds candidate edges using hotspotter and score them)
    query_object.exec_matching()
    query_object.apply_match_edges()
    query_object.apply_match_scores()
    # Use connected components to relabel names on nodes
    query_object.relabel_using_reviews()
    # Create a priority on edge review ands determines inconsistencies
    query_object.apply_review_inference()

    print('Precomputing match images')
    # view_orientation = request.args.get('view_orientation', 'vertical')
    precompute_current_review_match_images(ibs, query_object,
                                           global_feedback_limit=global_feedback_limit,
                                           view_orientation='vertical')
    # precompute_current_review_match_images(ibs, query_object,
    #                                        global_feedback_limit=global_feedback_limit,
    #                                        view_orientation='horizontal')
    return query_object


def load_identification_query_object(autoinit=False,
                                     global_feedback_limit=GLOBAL_FEEDBACK_LIMIT,
                                     **kwargs):
    ibs = current_app.ibs

    if current_app.QUERY_OBJECT is None:
        autoinit = True
    else:
        if current_app.QUERY_OBJECT.GLOBAL_FEEDBACK_COUNTER + 1 >= global_feedback_limit:
            # Apply current names that have been made to database
            commit_current_query_object_names(current_app.QUERY_OBJECT, ibs)

            # Rebuild the AnnotInference object via the engine
            autoinit = True

    if autoinit:
        query_object = _init_identification_query_object(ibs, **kwargs)

        # Assign to current_app's QUERY_OBJECT attribute
        query_object.GLOBAL_FEEDBACK_COUNTER = 0
        current_app.QUERY_OBJECT = query_object

    # Load query object and apply and buffered feedback before returning object
    query_object = current_app.QUERY_OBJECT
    while len(current_app.QUERY_OBJECT_FEEDBACK_BUFFER) > 0:
        feedback = current_app.QUERY_OBJECT_FEEDBACK_BUFFER.pop()
        print('Popping %r out of QUERY_OBJECT_FEEDBACK_BUFFER' % (feedback, ))
        aid1, aid2, state, tags = feedback
        query_object.add_feedback((aid1, aid2), decision=state, tags=tags)
        query_object.GLOBAL_FEEDBACK_COUNTER += 1

    return query_object


def check_engine_identification_query_object(global_feedback_limit=GLOBAL_FEEDBACK_LIMIT):
    ibs = current_app.ibs

    # We need to precompute the chips in this process for OpenCV, as warp-affine
    # segfaults in the web engine process
    # ibs.depc.get_rowids('chips', ibs.get_valid_aids())
    # ibs.depc.get_rowids('probchip', ibs.get_valid_aids())

    if current_app.QUERY_OBJECT is not None:
        if current_app.QUERY_OBJECT.GLOBAL_FEEDBACK_COUNTER + 1 >= global_feedback_limit:
            # Apply current names that have been made to database
            commit_current_query_object_names(ibs, current_app.QUERY_OBJECT)

            # Rebuild the AnnotInference object via the engine
            current_app.QUERY_OBJECT_JOBID = None

    if current_app.QUERY_OBJECT_JOBID is None:
        current_app.QUERY_OBJECT = None
        current_app.QUERY_OBJECT_JOBID = ibs.start_web_query_all()
        # import ibeis
        # web_ibs = ibeis.opendb_bg_web(dbdir=ibs.dbdir, port=6000)
        # query_object_jobid = web_ibs.send_ibeis_request('/api/engine/query/graph/')
        # print('query_object_jobid = %r' % (query_object_jobid, ))
        # current_app.QUERY_OBJECT_JOBID = query_object_jobid

    query_object_status_dict = ibs.get_job_status(current_app.QUERY_OBJECT_JOBID)
    args = (current_app.QUERY_OBJECT_JOBID, query_object_status_dict, )
    print('job id %r: %r' % args)

    if query_object_status_dict['jobstatus'] == 'completed':
        query_object_result = ibs.get_job_result(current_app.QUERY_OBJECT_JOBID)
        args = (current_app.QUERY_OBJECT_JOBID, query_object_result, )
        print('job id %r: %r' % args)
        assert query_object_result['json_result'] == 'precomputed'
        return True

    return False


@register_route('/turk/identification/', methods=['GET'])
def turk_identification(aid1=None, aid2=None, use_engine=False,
                        global_feedback_limit=GLOBAL_FEEDBACK_LIMIT,
                        **kwargs):
    """
    CommandLine:
        python -m ibeis.web.routes turk_identification --db PZ_Master1
        python -m ibeis.web.routes turk_identification --db PZ_MTEST
        python -m ibeis.web.routes turk_identification --db testdb1 --show

    Example:
        >>> # SCRIPT
        >>> from ibeis.other.ibsfuncs import *  # NOQA
        >>> import ibeis
        >>> web_ibs = ibeis.opendb_bg_web('testdb1')
        >>> resp = web_ibs.get('/turk/identification/')
        >>> web_ibs.terminate2()
        >>> ut.quit_if_noshow()
        >>> import plottool as pt
        >>> ut.render_html(resp.content)
        >>> ut.show_if_requested()
    """
    from ibeis.web import apis_query

    with ut.Timer('[web.routes.turk_identification] Load query_object'):
        ibs = current_app.ibs

        if use_engine:
            engine_computed = check_engine_identification_query_object(
                global_feedback_limit=global_feedback_limit
            )
        else:
            engine_computed = True

        if engine_computed:
            # ibs.depc_annot.get_rowids('chips', ibs.get_valid_aids())
            # ibs.depc_annot.get_rowids('probchip', ibs.get_valid_aids())

            query_object = load_identification_query_object(
                global_feedback_limit=global_feedback_limit
            )

            with ut.Timer('[web.routes.turk_identification] Get matches'):
                # Get raw list of reviews
                review_cfg = GLOBAL_FEEDBACK_CONFIG_DICT.copy()
                raw_review_list, _ = query_object.get_filtered_edges(review_cfg)

                # Get actual
                review_cfg['max_num'] = global_feedback_limit  # Controls the top X to be randomly sampled and displayed to all concurrent users
                review_aid1_list, review_aid2_list = query_object.get_filtered_edges(review_cfg)

            with ut.Timer('[web.routes.turk_identification] Get status'):
                # Get status
                try:
                    status_dict = query_object.connected_component_status()
                except RuntimeError:
                    status_dict = {}
                    status_dict['num_names_max'] = np.nan
                    status_dict['num_names_min'] = np.nan
                # status_remaining = status_dict['num_names_max'] - status_dict['num_names_min']
                status_remaining = status_dict['num_names_max'] - 0
                print('Feedback counter    = %r / %r' % (query_object.GLOBAL_FEEDBACK_COUNTER, GLOBAL_FEEDBACK_LIMIT, ))
                print('Status dict         = %r' % (status_dict, ))
                print('Raw list len        = %r' % (len(raw_review_list), ))
                print('len(query_aid_list) = %r' % (len(query_object.aids), ))
                print('Estimated remaining = %r' % (status_remaining, ))
                print('Reviews list len    = %r' % (len(review_aid1_list), ))
                try:
                    progress = '%0.02f' % (100.0 * (1.0 - (status_remaining / len(query_object.aids))), )
                except ZeroDivisionError:
                    progress = '100.0'

                # aid1 = request.args.get('aid1', None)
                # aid2 = request.args.get('aid2', None)
                replace_review_rowid = int(request.args.get('replace_review_rowid', -1))
                choice = aid1 is not None and aid2 is not None

            with ut.Timer('[web.routes.turk_identification] Process choice'):
                if choice or (len(review_aid1_list) > 0 and len(review_aid2_list) > 0):

                    with ut.Timer('[web.routes.turk_identification] ... Pick choice'):
                        finished = False
                        if not choice:
                            index = random.randint(0, len(review_aid1_list) - 1)
                            print('Picked random index = %r' % (index, ))
                            aid1 = review_aid1_list[index]
                            aid2 = review_aid2_list[index]

                        aid1 = int(aid1)
                        aid2 = int(aid2)
                        annot_uuid_1 = ibs.get_annot_uuids(aid1)
                        annot_uuid_2 = ibs.get_annot_uuids(aid2)

                    # lookup ChipMatch object
                    qreq_ = query_object.qreq_

                    with ut.Timer('[web.routes.turk_identification] ... Get scores'):
                        # Get score
                        # idx = cm.daid2_idx[aid2]
                        # match_score = cm.name_score_list[idx]
                        # match_score = cm.aid2_score[aid2]
                        graph_dict = query_object.graph.get_edge_data(aid1, aid2)
                        match_score = graph_dict.get('score', -1.0)

                    with ut.Timer('[web.routes.turk_identification] ... Make images'):
                        cm, aid1, aid2 = query_object.lookup_cm(aid1, aid2)
                        view_orientation = request.args.get('view_orientation', 'vertical')
                        with ut.Timer('[web.routes.turk_identification] ... ... Render images2'):
                            try:
                                image_matches = apis_query.ensure_review_image(
                                    ibs, aid2, cm, qreq_,
                                    view_orientation=view_orientation)
                            except KeyError as ex:
                                ut.printex(ex, 'Failed to make review image', tb=True,
                                           keys=['cm.qaid', 'aid1', 'aid2'],
                                           iswarning=True)
                            try:
                                image_clean = apis_query.ensure_review_image(
                                    ibs, aid2, cm, qreq_,
                                    view_orientation=view_orientation,
                                    draw_matches=False)
                            except KeyError:
                                image_clean = np.zeros((100, 100, 3), dtype=np.uint8)
                                ut.printex(ex, 'Failed to make fallback review image', tb=True,
                                           keys=['cm.qaid', 'aid1', 'aid2'])

                        with ut.Timer('[web.routes.turk_identification] ... ... Embed images'):
                            image_matches_src = appf.embed_image_html(image_matches)
                            image_clean_src = appf.embed_image_html(image_clean)

                    with ut.Timer('[web.routes.turk_identification] ... Process previous'):
                        # Get previous
                        previous = request.args.get('previous', None)
                        if previous is not None and ';' in previous:
                            previous = tuple(map(int, previous.split(';')))
                            assert len(previous) == 3
                        # print('Previous = %r' % (previous, ))
                        # print('replace_review_rowid  = %r' % (replace_review_rowid, ))
                else:
                    finished = True
                    progress = 100.0
                    aid1 = None
                    aid2 = None
                    annot_uuid_1 = None
                    annot_uuid_2 = None
                    image_clean_src = None
                    image_matches_src = None
                    previous = None
                    replace_review_rowid = None
                    view_orientation = None
                    match_score = None
        else:
            finished = 'engine'
            progress = 100.0
            aid1 = None
            aid2 = None
            annot_uuid_1 = None
            annot_uuid_2 = None
            image_clean_src = None
            image_matches_src = None
            previous = None
            replace_review_rowid = None
            view_orientation = None
            match_score = None

    session_counter = current_app.QUERY_OBJECT.GLOBAL_FEEDBACK_COUNTER
    session_limit = global_feedback_limit

    timedelta_str = 'Unknown'
    if aid1 is not None and aid2 is not None:
        unixtime_list = ibs.get_image_unixtime(ibs.get_annot_gids([aid1, aid2]))
        if -1.0 not in unixtime_list and 0.0 not in unixtime_list:
            timedelta = abs(unixtime_list[1] - unixtime_list[0])
            secs = 60.0
            mins = 60.0 * 60.0
            hrs  = 60.0 * 60.0 * 24.0
            days = 60.0 * 60.0 * 24.0 * 365.0

            if timedelta < secs:
                timedelta_str = '%0.2f seconds' % (timedelta, )
            elif timedelta < mins:
                timedelta /= secs
                timedelta_str = '%0.2f minutes' % (timedelta, )
            elif timedelta < hrs:
                timedelta /= mins
                timedelta_str = '%0.2f hours' % (timedelta, )
            elif timedelta < days:
                timedelta /= hrs
                timedelta_str = '%0.2f days' % (timedelta, )
            else:
                timedelta /= days
                timedelta_str = '%0.2f years' % (timedelta, )

    callback_url = url_for('submit_identification')
    return appf.template('turk', 'identification',
                         match_score=match_score,
                         image_clean_src=image_clean_src,
                         image_matches_src=image_matches_src,
                         aid1=aid1,
                         aid2=aid2,
                         progress=progress,
                         session_counter=session_counter,
                         session_limit=session_limit,
                         timedelta_str=timedelta_str,
                         finished=finished,
                         annot_uuid_1=str(annot_uuid_1),
                         annot_uuid_2=str(annot_uuid_2),
                         previous=previous,
                         replace_review_rowid=replace_review_rowid,
                         view_orientation=view_orientation,
                         callback_url=callback_url,
                         callback_method='POST',
                         EMBEDDED_CSS=None,
                         EMBEDDED_JAVASCRIPT=None)


@register_route('/turk/quality/', methods=['GET'])
def turk_quality(**kwargs):
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


@register_route('/turk/demographics/', methods=['GET'])
def turk_demographics(**kwargs):
    ibs = current_app.ibs
    imgsetid = request.args.get('imgsetid', '')
    imgsetid = None if imgsetid == 'None' or imgsetid == '' else int(imgsetid)

    gid_list = ibs.get_valid_gids(imgsetid=imgsetid)
    aid_list = ut.flatten(ibs.get_image_aids(gid_list))
    nid_list = ibs.get_annot_nids(aid_list)
    reviewed_list = appf.imageset_annot_demographics_processed(ibs, aid_list, nid_list)
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
    display_instructions = request.cookies.get('ia-demographics_instructions_seen', 1) == 0
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
        viewpoint_list = ibs.get_annot_viewpoints(name_aid_list)
        name_aid_combined_list = list(zip(
            name_aid_list,
            quality_list,
            quality_text_list,
            viewpoint_list,
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

    return appf.template('turk', 'demographics',
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
def group_review(**kwargs):
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
def api_root(**kwargs):
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
            if method not in rule_dict.keys(**kwargs):
                rule_dict[method] = []
            rule_dict[method].append((method, url, ))
    for method in rule_dict.keys(**kwargs):
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
def upload(**kwargs):
    return appf.template(None, 'upload')


@register_route('/dbinfo/', methods=['GET'])
def dbinfo(**kwargs):
    try:
        ibs = current_app.ibs
        dbinfo_str = ibs.get_dbinfo_str()
    except:
        dbinfo_str = ''
    dbinfo_str_formatted = '<pre>%s</pre>' % (dbinfo_str, )
    return dbinfo_str_formatted


@register_route('/counts/', methods=['GET'])
def wb_counts(**kwargs):
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
def wb_counts_alias1(**kwargs):
    return wb_counts()


@register_route('/gzgc/counts.jsp', methods=['GET'], __api_postfix_check__=False)
def wb_counts_alias2(**kwargs):
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
