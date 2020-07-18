# -*- coding: utf-8 -*-
"""
Dependencies: flask, tornado
"""
from __future__ import absolute_import, division, print_function
import random
import math
import simplejson as json
from flask import request, redirect, current_app, url_for
from wbia.control import controller_inject
from wbia import constants as const
from wbia.constants import KEY_DEFAULTS, SPECIES_KEY
from wbia.web import appfuncs as appf
from wbia.web import routes_ajax
import utool as ut
import numpy as np
import os

(print, rrr, profile) = ut.inject2(__name__)

CLASS_INJECT_KEY, register_ibs_method = controller_inject.make_ibs_register_decorator(
    __name__
)
register_route = controller_inject.get_wbia_flask_route(__name__)


THROW_TEST_AOI_TURKING = False
THROW_TEST_AOI_TURKING_PERCENTAGE = 0.05
THROW_TEST_AOI_TURKING_ERROR_MODES = {
    'addition': [1, 2, 3],
    'deletion': [1, 2, 3],
    'alteration': [1, 2, 3],
    'translation': [1, 2, 3],
}


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


@register_route('/', methods=['GET'], __route_authenticate__=False)
def root(**kwargs):
    ibs = current_app.ibs

    dbname = ibs.dbname
    dbdir = ibs.dbdir

    if ibs.containerized:
        hostname = const.CONTAINER_NAME
    else:
        hostname = ibs.dbname

    embedded = dict(globals(), **locals())
    return appf.template(None, **embedded)


@register_route('/login/', methods=['GET'], __route_authenticate__=False)
def login(refer=None, *args, **kwargs):
    # Default refer
    if refer is None:
        refer = url_for('root')
    else:
        refer = appf.decode_refer_url(refer)

    # Prevent loops
    if refer.split('?')[0].strip('/') == url_for('login').strip('/'):
        refer = url_for('root')

    if controller_inject.authenticated():
        return redirect(refer)

    refer = appf.encode_refer_url(refer)

    organization_dict = {
        'rpi': (
            'RPI',
            [
                # 'jason.parham',
                'chuck.stewart',
                'hendrik.weideman',
                'angela.su',
                'joshua.beard',
            ],
        ),
        'uic': (
            'UIC',
            [
                'tanya.berger-wolf',
                'mosheh.berger-wolf',
                'ivan.brugere',
                'chainarong.amornbunchornvej',
                'jia.li',
                'aynaz.taheri',
                'james.searing',
                'lorenzo.semeria',
                'matteo.foglio',
                'eleonora.darnese ',
                'guido.muscioni',
                'riccardo.pressiani',
                'krishna.chandu',
                'mathew.yang',
                'nathan.seitz',
                'luis.love',
                'ashley.riley',
            ],
        ),
        'princeton': (
            'Princeton',
            [
                'dan.rubenstein',
                'kaia.tombak',
                'andrew.gersick',
                'ryan.herbert',
                'joe.bakcoleman',
                'lisa.pbryan',
                'jackie.kariithi',
                'megan.mcsherry',
                'qing.cao',
            ],
        ),
        'wildme': (
            'Wild Me',
            [
                'jason.holmberg',
                'jason.parham',
                'jon.vanoast',
                'colin.kingen',
                'drew.blount',
            ],
        ),
        'kitware': ('Kitware', ['jon.crall'],),
        'ctrl-h': ('CTRL-H', ['jon.hannis', 'melinda.hannis'],),
    }
    organization_dict_json = json.dumps(organization_dict)

    embedded = dict(globals(), **locals())
    return appf.template(None, 'login', **embedded)


@register_route('/logout/', methods=['GET'], __route_authenticate__=False)
def logout(**kwargs):
    controller_inject.deauthenticate()
    return redirect(url_for('root'))


@register_route('/view/', methods=['GET'])
def view(**kwargs):
    ibs = current_app.ibs

    ibs.update_all_image_special_imageset()
    imgsetids_list = ibs.get_valid_imgsetids()
    gid_list = ibs.get_valid_gids()
    aid_list = ibs.get_valid_aids()
    pid_list = ibs.get_valid_part_rowids()
    nid_list = ibs.get_valid_nids()

    return appf.template(
        'view',
        num_imgsetids=len(imgsetids_list),
        num_gids=len(gid_list),
        num_aids=len(aid_list),
        num_pids=len(pid_list),
        num_nids=len(nid_list),
    )


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
        species_tag: ibs.get_species_nice(species_rowid)
        for species_tag, species_rowid in list(zip(species_tag_list, species_rowid_list))
    }

    viewpoint_dict = {}
    for species, viewpoint in list(zip(species_list, viewpoint_list)):
        if species in species_set:
            if species not in viewpoint_dict:
                viewpoint_dict[species] = {}
            if viewpoint not in viewpoint_dict[species]:
                viewpoint_dict[species][viewpoint] = 0
            viewpoint_dict[species][viewpoint] += 1

    viewpoint_tag_list = const.VIEWTEXT_TO_VIEWPOINT_RADIANS.keys()
    pie_label_list = [str(const.VIEWPOINTALIAS_NICE[_]) for _ in viewpoint_tag_list]
    pie_values_list = [
        (
            species_nice_dict[species],
            [viewpoint_dict[species][_] for _ in viewpoint_tag_list],
        )
        for species in species_tag_list
    ]

    viewpoint_order_list = [
        'left',
        'frontleft',
        'front',
        'frontright',
        'right',
        'backright',
        'back',
        'backleft',
    ]
    pie_left_list = [
        'left',
        'frontleft',
        'front',
        'frontright',
        'right',
        'backright',
        'back',
        'backleft',
    ]
    pie_right_list = [
        'right',
        'backright',
        'back',
        'backleft',
        'left',
        'frontleft',
        'front',
        'frontright',
    ]
    viewpoint_tag_dict = {
        'zebra_grevys': pie_right_list,
        'zebra_plains': pie_left_list,
        'giraffe_masai': pie_left_list,
    }
    pie_label_corrected_list = list(
        map(str, ['Correct', '+45', '+90', '+135', '+180', '+225', '+270', '+315'])
    )
    pie_values_corrected_list = [
        (
            species_nice_dict[species],
            [viewpoint_dict[species][_] for _ in viewpoint_tag_dict[species]],
        )
        for species in species_tag_list
    ]

    # Get number of annotations per name as a histogram for each species
    embedded = dict(globals(), **locals())
    return appf.template('view', 'viewpoints', __wrapper_header__=False, **embedded)


@register_route('/view/advanced/0/', methods=['GET'])
def view_advanced0(**kwargs):
    def _date_list(gid_list):
        unixtime_list = ibs.get_image_unixtime(gid_list)
        datetime_list = [
            ut.unixtime_to_datetimestr(unixtime) if unixtime is not None else 'UNKNOWN'
            for unixtime in unixtime_list
        ]
        datetime_split_list = [datetime.split(' ') for datetime in datetime_list]
        date_list = [
            datetime_split[0] if len(datetime_split) == 2 else 'UNKNOWN'
            for datetime_split in datetime_split_list
        ]
        return date_list

    def filter_annots_imageset(aid_list):
        try:
            imgsetid = request.args.get('imgsetid', '')
            imgsetid = int(imgsetid)
            imgsetid_list = ibs.get_valid_imgsetids()
            assert imgsetid in imgsetid_list
        except Exception:
            print('ERROR PARSING IMAGESET ID FOR ANNOTATION FILTERING')
            return aid_list
        imgsetids_list = ibs.get_annot_imgsetids(aid_list)
        aid_list = [
            aid
            for aid, imgsetid_list_ in list(zip(aid_list, imgsetids_list))
            if imgsetid in imgsetid_list_
        ]
        return aid_list

    def filter_images_imageset(gid_list):
        try:
            imgsetid = request.args.get('imgsetid', '')
            imgsetid = int(imgsetid)
            imgsetid_list = ibs.get_valid_imgsetids()
            assert imgsetid in imgsetid_list
        except Exception:
            print('ERROR PARSING IMAGESET ID FOR IMAGE FILTERING')
            return gid_list
        imgsetids_list = ibs.get_image_imgsetids(gid_list)
        gid_list = [
            gid
            for gid, imgsetid_list_ in list(zip(gid_list, imgsetids_list))
            if imgsetid in imgsetid_list_
        ]
        return gid_list

    def filter_names_imageset(nid_list):
        try:
            imgsetid = request.args.get('imgsetid', '')
            imgsetid = int(imgsetid)
            imgsetid_list = ibs.get_valid_imgsetids()
            assert imgsetid in imgsetid_list
        except Exception:
            print('ERROR PARSING IMAGESET ID FOR ANNOTATION FILTERING')
            return nid_list
        aids_list = ibs.get_name_aids(nid_list)
        imgsetids_list = [
            set(ut.flatten(ibs.get_annot_imgsetids(aid_list))) for aid_list in aids_list
        ]
        nid_list = [
            nid
            for nid, imgsetid_list_ in list(zip(nid_list, imgsetids_list))
            if imgsetid in imgsetid_list_
        ]
        return nid_list

    def filter_annots_general(ibs, aid_list):
        if ibs.dbname == 'GGR-IBEIS':
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
        else:
            assert ibs.dbname == 'GGR2-IBEIS'
            # aid_list = ibs.check_ggr_valid_aids(aid_list, species='zebra_grevys', threshold=0.75)
            aid_list = ibs.check_ggr_valid_aids(
                aid_list, species='giraffe_reticulated', threshold=0.75
            )

        return aid_list

    ibs = current_app.ibs

    aid_list = ibs.get_valid_aids()
    aid_list = filter_annots_general(ibs, aid_list)
    # aid_list = filter_annots_imageset(aid_list)
    gid_list = ibs.get_annot_gids(aid_list)
    unixtime_list = ibs.get_image_unixtime(gid_list)
    nid_list = ibs.get_annot_name_rowids(aid_list)
    date_list = _date_list(gid_list)

    # flagged_date_list = None
    flagged_date_list = [
        '2015/03/01',
        '2015/03/02',
        '2016/01/30',
        '2016/01/31',
        '2018/01/27',
        '2018/01/28',
    ]

    gid_list_unique = list(set(gid_list))
    date_list_unique = _date_list(gid_list_unique)
    date_taken_dict = {}
    for gid, date in list(zip(gid_list_unique, date_list_unique)):
        if flagged_date_list is not None and date not in flagged_date_list:
            continue
        if date not in date_taken_dict:
            date_taken_dict[date] = [0, 0]
        date_taken_dict[date][1] += 1

    gid_list_all = ibs.get_valid_gids()
    gid_list_all = filter_images_imageset(gid_list_all)
    date_list_all = _date_list(gid_list_all)
    for gid, date in list(zip(gid_list_all, date_list_all)):
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
    for index, (unixtime, aid, nid, date) in enumerate(
        sorted(zip(unixtime_list, aid_list, nid_list, date_list))
    ):
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
    #     for func, opts in list(zip(optimization_funcs, regressed_opts))
    # ]
    # index_list = list(domain)
    prediction_list = []

    date_seen_dict.pop('UNKNOWN', None)
    bar_label_list = sorted(date_seen_dict.keys())
    bar_value_list1 = [date_taken_dict[date][0] for date in bar_label_list]
    bar_value_list2 = [date_taken_dict[date][1] for date in bar_label_list]
    bar_value_list3 = [date_seen_dict[date][0] for date in bar_label_list]
    bar_value_list4 = [date_seen_dict[date][1] for date in bar_label_list]
    bar_value_list5 = [date_seen_dict[date][2] for date in bar_label_list]
    bar_value_list6 = [date_seen_dict[date][3] for date in bar_label_list]

    # label_list += ['Models'] + [''] * (len(index_list) - len(label_list) - 1)
    # value_list += [0] * (len(index_list) - len(value_list))

    # Counts
    imgsetid_list = ibs.get_valid_imgsetids()
    gid_list = ibs.get_valid_gids()
    gid_list = filter_images_imageset(gid_list)
    aid_list = ibs.get_valid_aids()
    # aid_list = filter_annots_imageset(aid_list)
    nid_list = ibs.get_valid_nids()
    nid_list = filter_names_imageset(nid_list)
    # contributor_list = ibs.get_valid_contributor_rowids()
    note_list = ibs.get_image_notes(gid_list)
    note_list = [','.join(note.split(',')[:-1]) for note in note_list]
    contributor_list = set(note_list)
    # nid_list = ibs.get_valid_nids()
    aid_list_count = filter_annots_general(ibs, aid_list)
    # aid_list_count = filter_annots_imageset(aid_list_count)
    gid_list_count = list(set(ibs.get_annot_gids(aid_list_count)))
    nid_list_count_dup = ibs.get_annot_name_rowids(aid_list_count)
    nid_list_count = list(set(nid_list_count_dup))

    # Calculate the Petersen-Lincoln index form the last two days
    from wbia.other import dbinfo as dbinfo_

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
        for nid, gps in list(zip(nid_list_count_dup, gps_list_markers)):
            if gps[0] == -1.0 and gps[1] == -1.0:
                continue
            nid_track_dict[nid].append(gps)
        gps_list_tracks = [nid_track_dict[nid] for nid in sorted(nid_track_dict.keys())]
    else:
        __nid_list, gps_track_list, aid_track_list = ibs.get_name_gps_tracks(
            aid_list=aid_list_count
        )
        gps_list_tracks = list(map(lambda x: list(map(list, x)), gps_track_list))

    gps_list_markers = [gps for gps in gps_list_markers]
    gps_list_markers_tuple_all = [
        (gps, gid) for gps, gid in list(zip(gps_list_markers_all, gid_list))
    ]
    gps_list_markers_all = [_[0] for _ in gps_list_markers_tuple_all]
    gid_list_markers_all = [_[1] for _ in gps_list_markers_tuple_all]
    gps_list_tracks = [
        [gps for gps in gps_list_track] for gps_list_track in gps_list_tracks
    ]

    ALLOW_IMAGE_DATE_COLOR = False
    VERSION = 1
    # Colors for GPS
    color_none = [0, '#777777']
    color_day1 = [1, '#CA4141']
    color_day2 = [2, '#428BCA']
    color_resight = [3, '#9A41CA']
    combined_list_markers_all = []
    dataset_color_dict = {}
    if VERSION == 1:
        dataset_color_label_list = ['Day 1 Only', 'Resightings', 'Day 2 Only']
    else:
        dataset_color_label_list = ['Day 1 Only', 'Resightings', 'Day 2 Only', 'Unused']

    for gid, gps in list(zip(gid_list_markers_all, gps_list_markers_all)):
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

                if ALLOW_IMAGE_DATE_COLOR:
                    if current_date in ['2016/01/30', '2018/01/27']:
                        color = color_day1
                    elif current_date in ['2016/01/31', '2018/01/28']:
                        color = color_day2
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
                elif current_date == '2018/01/27':
                    if '2018/01/28' in date_list:
                        color = color_resight
                    else:
                        color = color_day1
                elif current_date == '2018/01/28':
                    if '2018/01/27' in date_list:
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
            gps
            if color_ == color_none[1]
            else [
                gps[0] + random.uniform(-JITTER_AMOUNT, JITTER_AMOUNT),
                gps[1] + random.uniform(-JITTER_AMOUNT, JITTER_AMOUNT),
            ]
            for gps, color_ in list(zip(gps_list_markers_all, color_list_markers_all))
        ]

    valid_aids = ibs.get_valid_aids()
    # valid_aids = filter_annots_imageset(valid_aids)
    used_gids = list(set(ibs.get_annot_gids(valid_aids)))
    # used_contributor_tags = list(set( ibs.get_image_contributor_tag(used_gids) ))
    note_list = ibs.get_image_notes(used_gids)
    note_list = [','.join(note.split(',')[:-1]) for note in note_list]
    used_contributor_tags = set(note_list)

    # Get Age and sex (By Annot)
    # annot_sex_list = ibs.get_annot_sex(valid_aids_)
    # annot_age_months_est_min = ibs.get_annot_age_months_est_min(valid_aids_)
    # annot_age_months_est_max = ibs.get_annot_age_months_est_max(valid_aids_)
    # age_list = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    # for sex, min_age, max_age in list(zip(annot_sex_list, annot_age_months_est_min, annot_age_months_est_max)):
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
    for nid, sex, min_ages, max_ages in list(
        zip(
            nid_list_count,
            name_sex_list,
            name_age_months_est_mins_list,
            name_age_months_est_maxs_list,
        )
    ):
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
        elif 36 <= min_age and (max_age is None or 36 <= max_age):
            age_list[sex][3] += 1

    DEMOGRAPHICS_INCLUDE_UNREVIEWED_AMBIGOUS = False
    age_total = sum(map(sum, age_list))
    if DEMOGRAPHICS_INCLUDE_UNREVIEWED_AMBIGOUS:
        age_total += age_unreviewed + age_ambiguous
    age_total = np.nan if age_total == 0 else age_total

    def age_fmt_str(x):
        return '% 4d (% 2.02f%%)' % (x, 100 * x / age_total,)

    age_str_list = [[age_fmt_str(age) for age in age_list_] for age_list_ in age_list]
    age_str_list.append(age_fmt_str(age_unreviewed))
    age_str_list.append(age_fmt_str(age_ambiguous))

    # dbinfo_str = dbinfo()
    dbinfo_str = 'SKIPPED DBINFO'

    path_dict = ibs.compute_ggr_path_dict()
    if 'North' in path_dict:
        path_dict.pop('North')
    if 'Core' in path_dict:
        path_dict.pop('Core')

    return appf.template(
        'view',
        'advanced0',
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
        __wrapper_header__=False,
    )


@register_route('/view/advanced/1/', methods=['GET'])
def view_advanced1(**kwargs):
    ibs = current_app.ibs

    species_tag_list = ['zebra_grevys', 'zebra_plains', 'giraffe_masai']
    species_rowid_list = ibs.get_species_rowids_from_text(species_tag_list)
    species_set = set(species_tag_list)
    species_nice_dict = {
        species_tag: ibs.get_species_nice(species_rowid)
        for species_tag, species_rowid in list(zip(species_tag_list, species_rowid_list))
    }

    aid_list = ibs.get_valid_aids()
    species_list = ibs.get_annot_species_texts(aid_list)
    viewpoint_list = ibs.get_annot_viewpoints(aid_list)
    viewpoint_dict = {}

    for species, viewpoint in list(zip(species_list, viewpoint_list)):
        if species in species_set:
            if species not in viewpoint_dict:
                viewpoint_dict[species] = {}
            if viewpoint not in viewpoint_dict[species]:
                viewpoint_dict[species][viewpoint] = 0
            viewpoint_dict[species][viewpoint] += 1

    viewpoint_tag_list = const.VIEWTEXT_TO_VIEWPOINT_RADIANS.keys()
    pie_label_list = [str(const.VIEWPOINTALIAS_NICE[_]) for _ in viewpoint_tag_list]
    pie_values_list = [
        (
            species_nice_dict[species],
            [viewpoint_dict[species][_] for _ in viewpoint_tag_list],
        )
        for species in species_tag_list
    ]

    pie_left_list = [
        'left',
        'frontleft',
        'front',
        'frontright',
        'right',
        'backright',
        'back',
        'backleft',
    ]
    pie_right_list = [
        'right',
        'backright',
        'back',
        'backleft',
        'left',
        'frontleft',
        'front',
        'frontright',
    ]
    viewpoint_tag_dict = {
        'zebra_grevys': pie_right_list,
        'zebra_plains': pie_left_list,
        'giraffe_masai': pie_left_list,
    }
    pie_label_corrected_list = list(
        map(str, ['Correct', '+45', '+90', '+135', '+180', '+225', '+270', '+315'])
    )
    pie_values_corrected_list = [
        (
            species_nice_dict[species],
            [viewpoint_dict[species][_] for _ in viewpoint_tag_dict[species]],
        )
        for species in species_tag_list
    ]

    gid_list = ibs.get_valid_gids()
    note_list = ibs.get_image_notes(gid_list)
    aids_list = ibs.get_image_aids(gid_list)
    viewpoints_list = ut.unflat_map(ibs.get_annot_viewpoints, aids_list)
    dataset_tag_list = ['GGR', 'GZGC']
    pie_label_images_list = ['Correct Viewpoint', '+/- 45 Viewpoint', 'Unused']
    pie_values_images_dict = {dataset_tag: [0, 0, 0] for dataset_tag in dataset_tag_list}
    allowed_viewpoint_dict = {
        'GGR': ['right', 'frontright', 'backright'],
        'GZGC': ['left', 'frontleft', 'backleft'],
    }
    for note, viewpoint_list in list(zip(note_list, viewpoints_list)):
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

    pie_values_images_list = [(_, pie_values_images_dict[_]) for _ in dataset_tag_list]

    nid_list = ibs.get_valid_nids()
    aids_list = ibs.get_name_aids(nid_list)
    species_list = map(
        list, map(set, ut.unflat_map(ibs.get_annot_species_texts, aids_list))
    )
    species_list = [None if len(_) != 1 else _[0] for _ in species_list]

    num_bins = 15
    histogram_dict = {}
    for nid, aids, species in list(zip(nid_list, aids_list, species_list)):
        if species not in histogram_dict:
            histogram_dict[species] = {}
        count = len(aids)
        count = min(count, num_bins)
        if count not in histogram_dict[species]:
            histogram_dict[species][count] = 0
        histogram_dict[species][count] += 1

    histogram_bins = list(range(1, num_bins + 1))
    bar_label_list = [
        '%s+' % (bin_,) if bin_ == histogram_bins[-1] else '%s' % (bin_,)
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
    return appf.template('view', 'advanced1', __wrapper_header__=False, **embedded)


@register_route('/view/advanced/2/', methods=['GET'])
def view_advanced2(**kwargs):
    def _date_list(gid_list):
        unixtime_list = ibs.get_image_unixtime(gid_list)
        datetime_list = [
            ut.unixtime_to_datetimestr(unixtime) if unixtime is not None else 'UNKNOWN'
            for unixtime in unixtime_list
        ]
        datetime_split_list = [datetime.split(' ') for datetime in datetime_list]
        date_list = [
            datetime_split[0] if len(datetime_split) == 2 else 'UNKNOWN'
            for datetime_split in datetime_split_list
        ]
        return date_list

    def filter_annots_imageset(aid_list):
        try:
            imgsetid = request.args.get('imgsetid', '')
            imgsetid = int(imgsetid)
            imgsetid_list = ibs.get_valid_imgsetids()
            assert imgsetid in imgsetid_list
        except Exception:
            print('ERROR PARSING IMAGESET ID FOR ANNOTATION FILTERING')
            return aid_list
        imgsetids_list = ibs.get_annot_imgsetids(aid_list)
        aid_list = [
            aid
            for aid, imgsetid_list_ in list(zip(aid_list, imgsetids_list))
            if imgsetid in imgsetid_list_
        ]
        return aid_list

    def filter_annots_general(ibs, aid_list):
        if ibs.dbname == 'GGR-IBEIS':
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
        else:
            assert ibs.dbname == 'GGR2-IBEIS'
            aid_list = ibs.check_ggr_valid_aids(
                aid_list, species='zebra_grevys', threshold=0.75
            )

        return aid_list

    ibs = current_app.ibs

    aid_list = ibs.get_valid_aids()
    aid_list = filter_annots_general(ibs, aid_list)
    # aid_list = filter_annots_imageset(aid_list)
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
    for unixtime, aid, nid, date, species in sorted(
        list(zip(unixtime_list, aid_list, nid_list, date_list, species_list))
    ):
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
    return appf.template('view', 'advanced2', __wrapper_header__=False, **embedded)


@register_route('/view/advanced/3/', methods=['GET'])
def view_advanced3(**kwargs):

    ibs = current_app.ibs

    gid_list = ibs.get_valid_gids()
    # gid_list = gid_list[:100] + gid_list[-100:]
    note_list = ibs.get_image_notes(gid_list)

    contrib_dict = {}
    skipped = 0
    for gid, note in list(zip(gid_list, note_list)):
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
            car = car.split(' ')[-1].strip("'")
            letter = letter.split(' ')[-1].strip("'")

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
    return appf.template('view', 'advanced3', __wrapper_header__=False, **embedded)


@register_route('/view/advanced/4/', methods=['GET'])
def view_advanced4(**kwargs):
    def _date_list(gid_list):
        unixtime_list = ibs.get_image_unixtime(gid_list)
        datetime_list = [
            ut.unixtime_to_datetimestr(unixtime) if unixtime is not None else 'UNKNOWN'
            for unixtime in unixtime_list
        ]
        datetime_split_list = [datetime.split(' ') for datetime in datetime_list]
        date_list = [
            datetime_split[0] if len(datetime_split) == 2 else 'UNKNOWN'
            for datetime_split in datetime_split_list
        ]
        return date_list

    def filter_species_of_interest(gid_list):
        wanted_set = set(['zebra_plains', 'zebra_grevys', 'giraffe_masai'])
        aids_list = ibs.get_image_aids(gid_list)
        speciess_list = ut.unflat_map(ibs.get_annot_species_texts, aids_list)
        speciess_set = map(set, speciess_list)
        gid_list_filtered = []
        for gid, species_set in list(zip(gid_list, speciess_set)):
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
        for gid, viewpoint_set in list(zip(gid_list, viewpoints_list)):
            intersect_list = list(wanted_set & viewpoint_set)
            if len(intersect_list) > 0:
                gid_list_filtered.append(gid)
        return gid_list_filtered

    def filter_bad_metadata(gid_list):
        wanted_set = set(['2015/03/01', '2015/03/02', '2016/01/30', '2016/01/31'])
        date_list = _date_list(gid_list)
        gps_list = ibs.get_image_gps(gid_list)
        gid_list_filtered = []
        for gid, date, gps in list(zip(gid_list, date_list, gps_list)):
            if date in wanted_set and gps != (-1.0, -1.0):
                gid_list_filtered.append(gid)
        return gid_list_filtered

    def filter_bad_quality(gid_list, allowed_quality_list):
        aids_list = ibs.get_image_aids(gid_list)
        wanted_set = set(allowed_quality_list)
        qualities_list = ut.unflat_map(ibs.get_annot_quality_texts, aids_list)
        qualities_list = map(set, qualities_list)
        gid_list_filtered = []
        for gid, quality_list in list(zip(gid_list, qualities_list)):
            intersect_list = list(wanted_set & quality_list)
            if len(intersect_list) > 0:
                gid_list_filtered.append(gid)
        return gid_list_filtered

    # def filter_singletons(gid_list):
    #     aids_list = ibs.get_image_aids(gid_list)
    #     nids_list = ut.unflat_map(ibs.get_annot_nids, aids_list)
    #     gid_list_filtered = []
    #     for gid, nid_list in list(zip(gid_list, nids_list)):
    #         print(gid)
    #         print(nid_list)
    #         aids_list_ = ibs.get_name_aids(nid_list)
    #         print(aids_list_)
    #         single = True
    #         for nid, aid_list in list(zip(nid_list, aids_list_)):
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
    for gid, note in list(zip(gid_list, note_list)):
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
    dataset_dict['GZGC'] = filter_viewpoints_of_interest(
        dataset_dict['GZGC'], allowed_viewpoint_dict['GZGC']
    )
    dataset_dict['GGR'] = filter_viewpoints_of_interest(
        dataset_dict['GGR'], allowed_viewpoint_dict['GGR']
    )

    num_viewpoint_gzgc = len(dataset_dict['GZGC'])
    num_viewpoint_ggr = len(dataset_dict['GGR'])

    print('viewpoint', num_viewpoint_gzgc)
    print('viewpoint', num_viewpoint_ggr)

    allowed_quality_dict = {
        'GGR': ['good', 'perfect'],
        'GZGC': ['ok', 'good', 'perfect'],
    }
    dataset_dict['GZGC'] = filter_bad_quality(
        dataset_dict['GZGC'], allowed_quality_dict['GZGC']
    )
    dataset_dict['GGR'] = filter_bad_quality(
        dataset_dict['GGR'], allowed_quality_dict['GZGC']
    )

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
        stage_list_ggr[i] - stage_list_ggr[i + 1] for i in range(len(stage_list_ggr) - 1)
    ]

    bar_value_list = map(list, list(zip(diff_list_gzgc, diff_list_ggr)))

    # Get number of annotations per name as a histogram for each species
    embedded = dict(globals(), **locals())
    return appf.template('view', 'advanced4', __wrapper_header__=False, **embedded)


@register_route('/view/map/', methods=['GET'])
def view_map(**kwargs):

    ibs = current_app.ibs

    gid_list = ibs.get_valid_gids()
    gps_list_ = list(map(list, ibs.get_image_gps(gid_list)))

    gps_list = []
    color_list = []

    color = '#CA4141'

    JITTER_GPS = True
    JITTER_AMOUNT = 0.0001
    for gid, gps in list(zip(gid_list, gps_list_)):

        if gps is None or gps == [-1, -1]:
            continue

        if JITTER_GPS:
            gps[0] + random.uniform(-JITTER_AMOUNT, JITTER_AMOUNT)
            gps[1] + random.uniform(-JITTER_AMOUNT, JITTER_AMOUNT)

        gps_list.append(gps)
        color_list.append(color)

    path_dict = ibs.compute_ggr_path_dict()

    if 'North' in path_dict:
        path_dict.pop('North')
    if 'Core' in path_dict:
        path_dict.pop('Core')

    embedded = dict(globals(), **locals())
    return appf.template('view', 'map', **embedded)


@register_route('/view/imagesets/', methods=['GET'])
def view_imagesets(**kwargs):
    ibs = current_app.ibs
    filtered = True
    ibs.update_all_image_special_imageset()
    imgsetid = request.args.get('imgsetid', '')
    if len(imgsetid) > 0:
        imgsetid_list = imgsetid.strip().split(',')
        imgsetid_list = [
            None if imgsetid_ == 'None' or imgsetid_ == '' else int(imgsetid_)
            for imgsetid_ in imgsetid_list
        ]
    else:
        imgsetid_list = ibs.get_valid_imgsetids()
        filtered = False
    imageset_text_list = ibs.get_imageset_text(imgsetid_list)
    start_time_posix_list = ibs.get_imageset_start_time_posix(imgsetid_list)
    datetime_list = [
        ut.unixtime_to_datetimestr(start_time_posix)
        if start_time_posix is not None
        else 'Unknown'
        for start_time_posix in start_time_posix_list
    ]

    ######################################################################################
    all_gid_list = ibs.get_valid_gids()
    all_aid_list = ibs.get_valid_aids()

    gids_list = [ibs.get_valid_gids(imgsetid=imgsetid_) for imgsetid_ in imgsetid_list]
    num_gids = list(map(len, gids_list))

    ######################################################################################
    all_gid_aids_list = ibs.get_image_aids(all_gid_list)
    gid_aid_mapping = {
        gid: aid_list for gid, aid_list in zip(all_gid_list, all_gid_aids_list)
    }

    aids_list = [
        ut.flatten([gid_aid_mapping.get(gid, []) for gid in gid_list])
        for gid_list in gids_list
    ]
    num_aids = list(map(len, aids_list))

    ######################################################################################
    all_gid_reviewed_list = appf.imageset_image_processed(ibs, all_gid_list)
    gid_reviewed_mapping = {
        gid: reviewed for gid, reviewed in zip(all_gid_list, all_gid_reviewed_list)
    }
    images_reviewed_list = [
        [gid_reviewed_mapping.get(gid, False) for gid in gid_list]
        for gid_list in gids_list
    ]

    ######################################################################################
    all_aid_reviewed_list = appf.imageset_annot_viewpoint_processed(ibs, all_aid_list)
    aid_reviewed_mapping = {
        aid: reviewed for aid, reviewed in zip(all_aid_list, all_aid_reviewed_list)
    }
    annots_reviewed_viewpoint_list = [
        [aid_reviewed_mapping.get(aid, False) for aid in aid_list]
        for aid_list in aids_list
    ]

    ######################################################################################
    all_aid_reviewed_list = appf.imageset_annot_quality_processed(ibs, all_aid_list)
    aid_reviewed_mapping = {
        aid: reviewed for aid, reviewed in zip(all_aid_list, all_aid_reviewed_list)
    }
    annots_reviewed_quality_list = [
        [aid_reviewed_mapping.get(aid, False) for aid in aid_list]
        for aid_list in aids_list
    ]

    ######################################################################################
    image_processed_list = [
        images_reviewed.count(True) for images_reviewed in images_reviewed_list
    ]
    annot_processed_viewpoint_list = [
        annots_reviewed.count(True) for annots_reviewed in annots_reviewed_viewpoint_list
    ]
    annot_processed_quality_list = [
        annots_reviewed.count(True) for annots_reviewed in annots_reviewed_quality_list
    ]
    reviewed_list = [
        all(images_reviewed)
        and all(annots_reviewed_viewpoint)
        and all(annot_processed_quality)
        for images_reviewed, annots_reviewed_viewpoint, annot_processed_quality in list(
            zip(
                images_reviewed_list,
                annots_reviewed_viewpoint_list,
                annots_reviewed_quality_list,
            )
        )
    ]
    is_normal_list = ut.not_list(ibs.is_special_imageset(imgsetid_list))

    imageset_list = list(
        zip(
            is_normal_list,
            imgsetid_list,
            imageset_text_list,
            num_gids,
            image_processed_list,
            num_aids,
            annot_processed_viewpoint_list,
            annot_processed_quality_list,
            start_time_posix_list,
            datetime_list,
            reviewed_list,
        )
    )
    imageset_list.sort(key=lambda t: t[0])

    return appf.template(
        'view',
        'imagesets',
        filtered=filtered,
        imgsetid_list=imgsetid_list,
        imgsetid_list_str=','.join(map(str, imgsetid_list)),
        num_imgsetids=len(imgsetid_list),
        imageset_list=imageset_list,
        num_imagesets=len(imageset_list),
    )


@register_route('/view/graphs/', methods=['GET'])
def view_graphs(sync=False, **kwargs):
    ibs = current_app.ibs

    if sync:
        import wbia
        import wbia.constants as const

        aid_list_ = ibs.get_valid_aids()

        # Save sex from current name
        sex_list_ = ibs.get_annot_sex(aid_list_)

        nid_list = [const.UNKNOWN_NAME_ROWID] * len(aid_list_)
        ibs.set_annot_name_rowids(aid_list_, nid_list, notify_wildbook=False)
        ibs.delete_empty_nids()

        nid_list = ibs.get_valid_nids()
        print('Delete nids: %d' % (len(nid_list),))

        species_list = ['zebra_grevys', 'giraffe_reticulated']
        for species in species_list:
            assert ibs.dbname == 'GGR2-IBEIS'
            aid_list = ibs.check_ggr_valid_aids(
                aid_list_, species=species, threshold=0.75
            )
            ibs.set_annot_names_to_different_new_names(aid_list, notify_wildbook=False)

            infr = wbia.AnnotInference(ibs=ibs, aids=aid_list, autoinit=True)
            infr.reset_feedback('staging', apply=True)
            num_pccs = len(list(infr.positive_components()))
            print('\tproposing PCCs for %r: %d' % (species, num_pccs,))

            infr.relabel_using_reviews(rectify=True)
            infr.write_wbia_staging_feedback()
            infr.write_wbia_annotmatch_feedback()
            infr.write_wbia_name_assignment(notify_wildbook=False)

        ibs.delete_empty_nids()
        nid_list = ibs.get_valid_nids()
        print('Final nids: %d' % (len(nid_list),))

        # Reset sex for new names from old names
        nid_list_ = ibs.get_annot_nids(aid_list_)

        seen = set([])
        new_nid_list = []
        new_sex_list = []
        for nid_, sex_ in zip(nid_list_, sex_list_):
            if nid_ > 0 and nid_ is not None and nid_ not in seen:
                new_nid_list.append(nid_)
                new_sex_list.append(sex_)
                seen.add(nid_)
        ibs.set_name_sex(new_nid_list, new_sex_list)

    graph_uuid_list = current_app.GRAPH_CLIENT_DICT
    num_graphs = len(graph_uuid_list)

    graph_list = []
    for graph_uuid in graph_uuid_list:
        graph_client = current_app.GRAPH_CLIENT_DICT.get(graph_uuid, None)
        if graph_client is None:
            continue
        graph_uuid_str = 'graph_uuid=%s' % (ut.to_json(graph_uuid),)
        graph_uuid_str = graph_uuid_str.replace(': ', ':')
        graph_status, graph_exception = graph_client.refresh_status()
        infr_status = graph_client.infr_status
        if infr_status is None:
            infr_status = {}
        if graph_exception is not None:
            import traceback

            trace = traceback.format_tb(graph_exception.__traceback__)
            trace = '\n\n'.join(trace)
            graph_exception = 'Exception: %s\n%s' % (graph_exception, trace,)
        if graph_client.review_dict is None:
            num_edges = None
        else:
            edge_list = list(graph_client.review_dict.keys())
            num_edges = len(edge_list)
        phase = 'Phase %s (%s)' % (
            infr_status.get('phase', None),
            infr_status.get('loop_phase', None),
        )
        state = 0
        if infr_status.get('is_inconsistent', False):
            state = -1
        elif infr_status.get('is_converged', False):
            state = 1
        reviews = '%s (%s - %s)' % (
            num_edges,
            infr_status.get('num_meaningful', None),
            infr_status.get('num_pccs', None),
        )
        graph = (
            graph_uuid,
            graph_client.imagesets,
            graph_status,
            graph_exception,
            len(graph_client.aids),
            reviews,
            graph_uuid_str,
            phase,
            state,
        )
        graph_list.append(graph)

    embedded = dict(globals(), **locals())
    return appf.template('view', 'graphs', **embedded)


@register_route('/view/jobs/', methods=['GET'])
def view_jobs(**kwargs):
    ibs = current_app.ibs

    response = ibs.get_job_status()
    assert response['status'] == 'ok'

    jobs = response['json_result']
    job_list = []

    jobid_list = list(jobs.keys())
    job_values = [jobs[jobid] for jobid in jobid_list]
    jobnumber_list = ut.take_column(job_values, 'jobcounter')
    index_list = np.argsort(jobnumber_list)
    index_list = index_list[::-1]
    jobid_list_ = ut.take(jobid_list, index_list)

    for jobid in jobid_list_:
        job = jobs[jobid]
        jov_status = job['status']
        job_state = 0
        if jov_status == 'completed':
            job_state += 1
        if jov_status == 'exception':
            job_state -= 1
        if jov_status == 'corrupted':
            job_state -= 1
        job_exception = None
        job_list.append(
            (
                jobid,
                job['jobcounter'],
                job['action'],
                job['endpoint'],
                job['function'],
                jov_status,
                job_state,
                job['time_received'],
                job['time_started'],
                job['time_completed'],
                job['time_runtime'],
                job['time_turnaround'],
            )
        )

    num_jobs = len(job_list)

    embedded = dict(globals(), **locals())
    return appf.template('view', 'jobs', **embedded)


@register_route('/view/image/<gid>/', methods=['GET'])
def image_view_api(gid=None, thumbnail=False, fresh=False, **kwargs):
    r"""
    Returns the base64 encoded image of image <gid>

    RESTful:
        Method: GET
        URL:    /image/view/<gid>/
    """
    encoded = routes_ajax.image_src(gid, thumbnail=thumbnail, fresh=fresh)
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
        gid_list = [
            None if gid_ == 'None' or gid_ == '' else int(gid_) for gid_ in gid_list
        ]
    elif len(imgsetid) > 0:
        imgsetid_list = imgsetid.strip().split(',')
        imgsetid_list = [
            None if imgsetid_ == 'None' or imgsetid_ == '' else int(imgsetid_)
            for imgsetid_ in imgsetid_list
        ]
        gid_list = ut.flatten(
            [ibs.get_valid_gids(imgsetid=imgsetid) for imgsetid_ in imgsetid_list]
        )
    else:
        gid_list = ibs.get_valid_gids()
        filtered = False
    # Page
    gid_list = sorted(gid_list)
    page_start = min(len(gid_list), (page - 1) * appf.PAGE_SIZE)
    page_end = min(len(gid_list), page * appf.PAGE_SIZE)
    page_total = int(math.ceil(len(gid_list) / appf.PAGE_SIZE))
    page_previous = None if page_start == 0 else page - 1
    page_next = None if page_end == len(gid_list) else page + 1
    gid_list = gid_list[page_start:page_end]
    print(
        '[web] Loading Page [ %d -> %d ] (%d), Prev: %s, Next: %s'
        % (page_start, page_end, len(gid_list), page_previous, page_next,)
    )
    image_unixtime_list = ibs.get_image_unixtime(gid_list)
    datetime_list = [
        ut.unixtime_to_datetimestr(image_unixtime)
        if image_unixtime is not None
        else 'Unknown'
        for image_unixtime in image_unixtime_list
    ]
    image_list = list(
        zip(
            gid_list,
            [
                ','.join(map(str, imgsetid_list_))
                for imgsetid_list_ in ibs.get_image_imgsetids(gid_list)
            ],
            ibs.get_image_gnames(gid_list),
            image_unixtime_list,
            datetime_list,
            ibs.get_image_gps(gid_list),
            ibs.get_image_party_tag(gid_list),
            ibs.get_image_contributor_tag(gid_list),
            ibs.get_image_notes(gid_list),
            appf.imageset_image_processed(ibs, gid_list),
        )
    )
    # image_list.sort(key=lambda t: t[3])
    return appf.template(
        'view',
        'images',
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
        page_next=page_next,
    )


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
        aid_list = [
            None if aid_ == 'None' or aid_ == '' else int(aid_) for aid_ in aid_list
        ]
    elif len(gid) > 0:
        gid_list = gid.strip().split(',')
        gid_list = [
            None if gid_ == 'None' or gid_ == '' else int(gid_) for gid_ in gid_list
        ]
        aid_list = ut.flatten(ibs.get_image_aids(gid_list))
    elif len(imgsetid) > 0:
        imgsetid_list = imgsetid.strip().split(',')
        imgsetid_list = [
            None if imgsetid_ == 'None' or imgsetid_ == '' else int(imgsetid_)
            for imgsetid_ in imgsetid_list
        ]
        gid_list = ut.flatten(
            [ibs.get_valid_gids(imgsetid=imgsetid_) for imgsetid_ in imgsetid_list]
        )
        aid_list = ut.flatten(ibs.get_image_aids(gid_list))
    else:
        aid_list = ibs.get_valid_aids()
        filtered = False
    # Page
    page_start = min(len(aid_list), (page - 1) * appf.PAGE_SIZE)
    page_end = min(len(aid_list), page * appf.PAGE_SIZE)
    page_total = int(math.ceil(len(aid_list) / appf.PAGE_SIZE))
    page_previous = None if page_start == 0 else page - 1
    page_next = None if page_end == len(aid_list) else page + 1
    aid_list = aid_list[page_start:page_end]
    print(
        '[web] Loading Page [ %d -> %d ] (%d), Prev: %s, Next: %s'
        % (page_start, page_end, len(aid_list), page_previous, page_next,)
    )
    annotation_list = list(
        zip(
            aid_list,
            ibs.get_annot_gids(aid_list),
            [
                ','.join(map(str, imgsetid_list_))
                for imgsetid_list_ in ibs.get_annot_imgsetids(aid_list)
            ],
            ibs.get_annot_image_names(aid_list),
            ibs.get_annot_names(aid_list),
            ibs.get_annot_exemplar_flags(aid_list),
            ibs.get_annot_species_texts(aid_list),
            ibs.get_annot_viewpoints(aid_list),
            ibs.get_annot_quality_texts(aid_list),
            ibs.get_annot_sex_texts(aid_list),
            ibs.get_annot_age_months_est(aid_list),
            ibs.get_annot_reviewed(aid_list),
            ibs.get_annot_interest(aid_list),
            # [ reviewed_viewpoint and reviewed_quality for reviewed_viewpoint, reviewed_quality in list(zip(appf.imageset_annot_viewpoint_processed(ibs, aid_list), appf.imageset_annot_quality_processed(ibs, aid_list))) ],
        )
    )
    annotation_list.sort(key=lambda t: t[0])
    return appf.template(
        'view',
        'annotations',
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
        page_next=page_next,
    )


@register_route('/view/parts/', methods=['GET'])
def view_parts(
    pid_list=None, aid_list=None, gid_list=None, imgsetid_list=None, page=1, **kwargs
):
    ibs = current_app.ibs

    filtered = True
    if imgsetid_list is not None:
        gid_list = ut.flatten(ibs.get_imageset_gids(imgsetid_list))
    if gid_list is not None:
        aid_list = ut.flatten(ibs.get_image_aids(gid_list))
    if aid_list is not None:
        pid_list = ut.flatten(ibs.get_annot_part_rowids(aid_list))
    else:
        pid_list = ibs.get_valid_part_rowids()
        filtered = False

    # Page
    page_start = min(len(pid_list), (page - 1) * appf.PAGE_SIZE)
    page_end = min(len(pid_list), page * appf.PAGE_SIZE)
    page_total = int(math.ceil(len(pid_list) / appf.PAGE_SIZE))
    page_previous = None if page_start == 0 else page - 1
    page_next = None if page_end == len(pid_list) else page + 1

    pid_list = pid_list[page_start:page_end]
    print(
        '[web] Loading Page [ %d -> %d ] (%d), Prev: %s, Next: %s'
        % (page_start, page_end, len(pid_list), page_previous, page_next,)
    )

    pid_list = [] if pid_list is None else pid_list
    aid_list = [] if aid_list is None else aid_list
    gid_list = [] if gid_list is None else gid_list
    imgsetid_list = [] if imgsetid_list is None else imgsetid_list

    imgsetid_list_str = ','.join(map(str, imgsetid_list))
    gid_list_str = ','.join(map(str, gid_list))
    aid_list_str = ','.join(map(str, aid_list))
    pid_list_str = ','.join(map(str, pid_list))

    num_imgsetids = len(imgsetid_list)
    num_gids = len(gid_list)
    num_aids = len(aid_list)
    num_pids = len(pid_list)

    part_list = list(
        zip(
            pid_list,
            ibs.get_part_aids(pid_list),
            ibs.get_part_gids(pid_list),
            ibs.get_part_types(pid_list),
            ibs.get_part_reviewed(pid_list),
        )
    )
    part_list.sort(key=lambda t: t[0])

    embedded = dict(globals(), **locals())
    return appf.template('view', 'parts', **embedded)


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
        nid_list = [
            None if nid_ == 'None' or nid_ == '' else int(nid_) for nid_ in nid_list
        ]
    if len(aid) > 0:
        aid_list = aid.strip().split(',')
        aid_list = [
            None if aid_ == 'None' or aid_ == '' else int(aid_) for aid_ in aid_list
        ]
        nid_list = ibs.get_annot_name_rowids(aid_list)
    elif len(gid) > 0:
        gid_list = gid.strip().split(',')
        gid_list = [
            None if gid_ == 'None' or gid_ == '' else int(gid_) for gid_ in gid_list
        ]
        aid_list = ut.flatten(ibs.get_image_aids(gid_list))
        nid_list = ibs.get_annot_name_rowids(aid_list)
    elif len(imgsetid) > 0:
        imgsetid_list = imgsetid.strip().split(',')
        imgsetid_list = [
            None if imgsetid_ == 'None' or imgsetid_ == '' else int(imgsetid_)
            for imgsetid_ in imgsetid_list
        ]
        gid_list = ut.flatten(
            [ibs.get_valid_gids(imgsetid=imgsetid_) for imgsetid_ in imgsetid_list]
        )
        aid_list = ut.flatten(ibs.get_image_aids(gid_list))
        nid_list = ibs.get_annot_name_rowids(aid_list)
    else:
        nid_list = ibs.get_valid_nids()
        filtered = False
    # Page
    appf.PAGE_SIZE_ = int(appf.PAGE_SIZE / 5)
    page_start = min(len(nid_list), (page - 1) * appf.PAGE_SIZE_)
    page_end = min(len(nid_list), page * appf.PAGE_SIZE_)
    page_total = int(math.ceil(len(nid_list) / appf.PAGE_SIZE_))
    page_previous = None if page_start == 0 else page - 1
    page_next = None if page_end == len(nid_list) else page + 1
    nid_list = nid_list[page_start:page_end]
    print(
        '[web] Loading Page [ %d -> %d ] (%d), Prev: %s, Next: %s'
        % (page_start, page_end, len(nid_list), page_previous, page_next,)
    )
    aids_list = ibs.get_name_aids(nid_list)
    annotations_list = [
        list(
            zip(
                aid_list_,
                ibs.get_annot_gids(aid_list_),
                [
                    ','.join(map(str, imgsetid_list_))
                    for imgsetid_list_ in ibs.get_annot_imgsetids(aid_list_)
                ],
                ibs.get_annot_image_names(aid_list_),
                ibs.get_annot_names(aid_list_),
                ibs.get_annot_exemplar_flags(aid_list_),
                ibs.get_annot_species_texts(aid_list_),
                ibs.get_annot_viewpoints(aid_list_),
                ibs.get_annot_quality_texts(aid_list_),
                ibs.get_annot_sex_texts(aid_list_),
                ibs.get_annot_age_months_est(aid_list_),
                [
                    reviewed_viewpoint and reviewed_quality
                    for reviewed_viewpoint, reviewed_quality in list(
                        zip(
                            appf.imageset_annot_viewpoint_processed(ibs, aid_list_),
                            appf.imageset_annot_quality_processed(ibs, aid_list_),
                        )
                    )
                ],
            )
        )
        for aid_list_ in aids_list
    ]
    name_list = list(zip(nid_list, annotations_list))
    name_list.sort(key=lambda t: t[0])
    return appf.template(
        'view',
        'names',
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
        page_next=page_next,
    )


@register_route('/action/', methods=['GET'])
def action(**kwargs):
    ibs = current_app.ibs

    imgsetid = request.args.get('imgsetid', '')
    imgsetid = None if imgsetid == 'None' or imgsetid == '' else int(imgsetid)

    job_id_list = ibs.get_job_id_list()
    job_action_list = [None] * len(job_id_list)
    job_status_list = [ibs.get_job_status(job_id)['jobstatus'] for job_id in job_id_list]
    action_list = list(zip(job_id_list, job_action_list, job_status_list))

    return appf.template(
        'action',
        None,
        imgsetid=imgsetid,
        action_list=action_list,
        num_actions=len(action_list),
    )


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
        >>> from wbia.web.apis_detect import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb(defaultdb='testdb1')
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
            'left': bbox_percent[0],
            'top': bbox_percent[1],
            'width': bbox_percent[2],
            'height': bbox_percent[3],
            'label': annot_.species,
            'id': annot_.aids,
            'theta': annot_.thetas,
            'tags': annot_.case_tags,
        }
    annotation_list.append(temp)


@register_route('/turk/cameratrap/', methods=['GET'])
def turk_cameratrap(**kwargs):
    ibs = current_app.ibs
    tup = appf.get_turk_image_args(appf.imageset_image_cameratrap_processed)
    (gid_list, reviewed_list, imgsetid, progress, gid, previous) = tup

    finished = gid is None
    if not finished:
        image_src = routes_ajax.image_src(gid, resize=False)
        positive = ibs.get_image_cameratrap(gid) not in [False, None]
    else:
        image_src = None
        positive = False

    imagesettext = ibs.get_imageset_text(imgsetid)

    embedded = dict(globals(), **locals())
    return appf.template('turk', 'cameratrap', **embedded)


@register_ibs_method
def precompute_web_detection_thumbnails(ibs, gid_list=None, **kwargs):
    if gid_list is None:
        gid_list = ibs.get_valid_gids()

    # kwargs['thumbsize'] = max(
    #     int(appf.TARGET_WIDTH),
    #     int(appf.TARGET_HEIGHT),
    # )
    # kwargs['draw_annots'] = False
    # ibs.get_image_thumbpath(gid_list, ensure_paths=True, **kwargs)

    ibs.depc_image.get('web_src', gid_list, 'src', read_extern=False)


@register_ibs_method
def precompute_web_viewpoint_thumbnails(ibs, aid_list=None, **kwargs):
    if aid_list is None:
        aid_list = ibs.get_valid_aids()

    kwargs['dim_size'] = max(int(appf.TARGET_WIDTH), int(appf.TARGET_HEIGHT),)
    ibs._parallel_chips = True
    ibs.get_annot_chip_fpath(aid_list, ensure=True, config2_=kwargs)


@register_route('/turk/detection/', methods=['GET'])
def turk_detection(
    gid=None,
    only_aid=None,
    refer_aid=None,
    imgsetid=None,
    previous=None,
    previous_only_aid=None,
    staged_super=False,
    progress=None,
    **kwargs,
):

    ibs = current_app.ibs

    staged_reviews_required = 3

    default_list = [
        ('autointerest', False),
        ('interest_bypass', False),
        ('metadata', True),
        ('metadata_viewpoint', True),
        ('metadata_quality', True),
        ('metadata_flags', True),
        ('metadata_flags_aoi', True),
        ('metadata_flags_multiple', True),
        ('metadata_species', True),
        ('metadata_label', True),
        ('metadata_quickhelp', True),
        ('parts', True),
        ('modes_rectangle', True),
        ('modes_diagonal', True),
        ('modes_diagonal2', True),
        ('staged', False),
        ('canonical', False),
    ]

    config_kwargs = kwargs.get('config', {})
    config = {
        key: kwargs.get(key, config_kwargs.get(key, default))
        for key, default in default_list
    }
    config_str_list = [
        '%s=%s' % (key, 'true' if config[key] else 'false',) for key in config.keys()
    ]
    config_str = '&'.join(config_str_list)

    is_staged = config['staged']
    is_staged = is_staged and appf.ALLOW_STAGED

    is_canonical = config['canonical']

    imgsetid = None if imgsetid == '' or imgsetid == 'None' else imgsetid
    imagesettext = None if imgsetid is None else ibs.get_imageset_text(imgsetid)

    if not is_canonical:
        gid_list = ibs.get_valid_gids(imgsetid=imgsetid)
        reviewed_list = appf.imageset_image_processed(
            ibs, gid_list, is_staged=is_staged, reviews_required=staged_reviews_required
        )

        try:
            progress = '%0.2f' % (100.0 * reviewed_list.count(True) / len(gid_list),)
        except ZeroDivisionError:
            progress = '100.0'

        if is_staged:
            staged_progress = appf.imageset_image_staged_progress(
                ibs, gid_list, reviews_required=staged_reviews_required
            )
            staged_progress = '%0.2f' % (100.0 * staged_progress,)
        else:
            staged_progress = None

        if gid is None:
            gid_list_ = ut.filterfalse_items(gid_list, reviewed_list)
            if len(gid_list_) == 0:
                gid = None
            else:
                # gid = gid_list_[0]
                gid = random.choice(gid_list_)
    else:
        staged_progress = None

    finished = gid is None
    review = 'review' in request.args.keys()
    display_instructions = (
        False  # request.cookies.get('ia-detection_instructions_seen', 1) == 1
    )
    display_new_features = request.cookies.get('ia-detection_new_features_seen', 1) == 1
    display_species_examples = (
        False  # request.cookies.get('ia-detection_example_species_seen', 0) == 0
    )

    if not finished:
        # image_src = routes_ajax.image_src(gid, resize=False)
        image_src = ibs.depc_image.get('web_src', gid, 'src')

        width, height = ibs.get_image_sizes(gid)
        staged_user = controller_inject.get_user()
        if staged_user is None:
            staged_user = {}
        staged_user_id = staged_user.get('username', None)

        # Get annotations
        aid_list = ibs.get_image_aids(gid, is_staged=is_staged)

        if is_canonical:
            assert only_aid in aid_list, 'Specified only_aid is not in this image'
            aid_list = [only_aid]

        if is_staged:
            # Filter aids for current user
            annot_user_id_list = ibs.get_annot_staged_user_ids(aid_list)
            aid_list = [
                aid
                for aid, annot_user_id in zip(aid_list, annot_user_id_list)
                if staged_super or annot_user_id == staged_user_id
            ]

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
        zipped = list(
            zip(
                aid_list,
                annot_bbox_list,
                annot_theta_list,
                species_list,
                viewpoint_list,
                quality_list,
                multiple_list,
                interest_list,
            )
        )
        for (
            aid,
            annot_bbox,
            annot_theta,
            species,
            viewpoint,
            quality,
            multiple,
            interest,
        ) in zipped:
            if quality in [-1, None]:
                quality = 0
            elif quality <= 2:
                quality = 1
            elif quality > 2:
                quality = 2

            viewpoint1, viewpoint2, viewpoint3 = appf.convert_viewpoint_to_tuple(
                viewpoint
            )

            temp = {}
            temp['id'] = aid
            temp['left'] = 100.0 * (annot_bbox[0] / width)
            temp['top'] = 100.0 * (annot_bbox[1] / height)
            temp['width'] = 100.0 * (annot_bbox[2] / width)
            temp['height'] = 100.0 * (annot_bbox[3] / height)
            temp['theta'] = float(annot_theta)
            temp['viewpoint1'] = viewpoint1
            temp['viewpoint2'] = viewpoint2
            temp['viewpoint3'] = viewpoint3
            temp['quality'] = quality
            temp['multiple'] = 'true' if multiple == 1 else 'false'
            temp['interest'] = 'true' if interest == 1 else 'false'
            temp['species'] = species

            mapping_dict[aid] = len(annotation_list)
            annotation_list.append(temp)

        # Get parts
        part_rowid_list = ut.flatten(
            ibs.get_annot_part_rowids(aid_list, is_staged=is_staged)
        )

        if is_staged:
            # Filter part_rowids for current user
            part_user_id_list = ibs.get_part_staged_user_ids(part_rowid_list)
            part_rowid_list = [
                part_rowid
                for part_rowid, part_user_id in zip(part_rowid_list, part_user_id_list)
                if staged_super or part_user_id == staged_user_id
            ]

        part_aid_list = ibs.get_part_aids(part_rowid_list)
        part_bbox_list = ibs.get_part_bboxes(part_rowid_list)
        part_theta_list = ibs.get_part_thetas(part_rowid_list)
        part_viewpoint_list = ibs.get_part_viewpoints(part_rowid_list)
        part_quality_list = ibs.get_part_qualities(part_rowid_list)
        part_type_list = ibs.get_part_types(part_rowid_list)
        # Get annotation bounding boxes

        part_list = []
        zipped = list(
            zip(
                part_rowid_list,
                part_aid_list,
                part_bbox_list,
                part_theta_list,
                part_viewpoint_list,
                part_quality_list,
                part_type_list,
            )
        )
        for (
            part_rowid,
            part_aid,
            part_bbox,
            part_theta,
            part_viewpoint,
            part_quality,
            part_type,
        ) in zipped:

            if is_canonical:
                if part_type != appf.CANONICAL_PART_TYPE:
                    continue

            if part_quality in [-1, None]:
                part_quality = 0
            elif part_quality <= 2:
                part_quality = 1
            elif part_quality > 2:
                part_quality = 2

            viewpoint1, viewpoint2, viewpoint3 = appf.convert_viewpoint_to_tuple(
                part_viewpoint
            )

            temp = {}
            temp['id'] = part_rowid
            temp['parent'] = mapping_dict[part_aid]
            temp['left'] = 100.0 * (part_bbox[0] / width)
            temp['top'] = 100.0 * (part_bbox[1] / height)
            temp['width'] = 100.0 * (part_bbox[2] / width)
            temp['height'] = 100.0 * (part_bbox[3] / height)
            temp['theta'] = float(part_theta)
            temp['viewpoint1'] = viewpoint1
            temp['quality'] = part_quality
            temp['type'] = part_type
            part_list.append(temp)

        if len(species_list) > 0:
            species = max(
                set(species_list), key=species_list.count
            )  # Get most common species
        elif appf.default_species(ibs) is not None:
            species = appf.default_species(ibs)
        else:
            species = KEY_DEFAULTS[SPECIES_KEY]

        staged_aid_list = ibs.get_image_aids(gid, is_staged=True)
        staged_part_rowid_list = ut.flatten(
            ibs.get_annot_part_rowids(staged_aid_list, is_staged=True)
        )

        imgesetid_list = list(set(ibs.get_image_imgsetids(gid)) - set([imgsetid]))
        imagesettext_list = ibs.get_imageset_text(imgesetid_list)
        imagesettext_list = [_ for _ in imagesettext_list if not _.startswith('*')]
        imagesettext_list_str = ', '.join(imagesettext_list)
        original_filename = os.path.split(ibs.get_image_uris_original(gid))[1]
    else:
        species = None
        image_src = None
        annotation_list = []
        part_list = []
        staged_aid_list = []
        staged_part_rowid_list = []
        imagesettext_list_str = None
        original_filename = None

    staged_uuid_list = ibs.get_annot_staged_uuids(staged_aid_list)
    staged_user_id_list = ibs.get_annot_staged_user_ids(staged_aid_list)

    num_staged_aids = len(staged_aid_list)
    num_staged_part_rowids = len(staged_part_rowid_list)
    num_staged_sessions = len(set(staged_uuid_list))
    num_staged_users = len(set(staged_user_id_list))

    THROW_TEST_AOI_TURKING_MANIFEST = []
    THROW_TEST_AOI_TURKING_AVAILABLE = False
    if THROW_TEST_AOI_TURKING:
        if random.uniform(0.0, 1.0) <= THROW_TEST_AOI_TURKING_PERCENTAGE:
            THROW_TEST_AOI_TURKING_AVAILABLE = True
            annotation_list = list(annotation_list)
            part_list = list(part_list)

            key_list = list(THROW_TEST_AOI_TURKING_ERROR_MODES.keys())
            throw_test_aoi_turking_mode = random.choice(key_list)
            throw_test_aoi_turking_severity = random.choice(
                THROW_TEST_AOI_TURKING_ERROR_MODES[throw_test_aoi_turking_mode]
            )
            throw_test_aoi_turking_severity = min(
                throw_test_aoi_turking_severity, len(annotation_list)
            )
            args = (
                throw_test_aoi_turking_mode,
                throw_test_aoi_turking_severity,
            )
            print('throw_test_aoi_turking: %r mode with %r severity' % args)

            index_list = list(range(len(annotation_list)))
            random.shuffle(index_list)
            index_list = index_list[:throw_test_aoi_turking_severity]
            index_list = sorted(index_list, reverse=True)

            args = (
                throw_test_aoi_turking_mode,
                throw_test_aoi_turking_severity,
            )
            if throw_test_aoi_turking_mode == 'addition':

                for index in index_list:
                    width = random.uniform(0.0, 100.0)
                    height = random.uniform(0.0, 100.0)
                    left = random.uniform(0.0, 100.0 - width)
                    top = random.uniform(0.0, 100.0 - height)
                    species = random.choice(ibs.get_all_species_texts())
                    annotation = {
                        'id': None,
                        'top': top,
                        'left': left,
                        'width': width,
                        'height': height,
                        'theta': 0.0,
                        'species': species,
                        'quality': 0,
                        'interest': 'false',
                        'multiple': 'false',
                        'viewpoint1': -1,
                        'viewpoint2': -1,
                        'viewpoint3': -1,
                    }
                    annotation_list.append(annotation)
                    THROW_TEST_AOI_TURKING_MANIFEST.append(
                        {'action': 'addition', 'values': annotation}
                    )
            elif throw_test_aoi_turking_mode == 'deletion':
                if len(part_list) == 0:

                    for index in index_list:
                        annotation = annotation_list.pop(index)
                        THROW_TEST_AOI_TURKING_MANIFEST.append(
                            {'action': 'deletion', 'values': annotation}
                        )
                else:
                    THROW_TEST_AOI_TURKING_AVAILABLE = False
            elif throw_test_aoi_turking_mode == 'alteration':
                for index in index_list:
                    direction_list = ['left', 'right', 'up', 'down']
                    direction = random.choice(direction_list)
                    height = annotation_list[index]['height']
                    width = annotation_list[index]['width']
                    left = annotation_list[index]['left']
                    top = annotation_list[index]['top']

                    if direction == 'left':
                        delta = width * 0.25
                        left -= delta
                        width += delta
                    elif direction == 'right':
                        delta = width * 0.25
                        width += delta
                    elif direction == 'up':
                        delta = height * 0.25
                        top -= delta
                        height += delta
                    elif direction == 'down':
                        delta = height * 0.25
                        height += delta

                    annotation_list[index]['height'] = min(max(height, 0.0), 100.0)
                    annotation_list[index]['width'] = min(max(width, 0.0), 100.0)
                    annotation_list[index]['left'] = min(max(left, 0.0), 100.0)
                    annotation_list[index]['top'] = min(max(top, 0.0), 100.0)

                    THROW_TEST_AOI_TURKING_MANIFEST.append(
                        {
                            'action': 'alteration-%s' % (direction,),
                            'values': annotation_list[index],
                        }
                    )
            elif throw_test_aoi_turking_mode == 'translation':
                for index in index_list:
                    direction_list = ['left', 'right', 'up', 'down']
                    direction = random.choice(direction_list)
                    height = annotation_list[index]['height']
                    width = annotation_list[index]['width']
                    left = annotation_list[index]['left']
                    top = annotation_list[index]['top']

                    if direction == 'left':
                        delta = width * 0.25
                        left -= delta
                    elif direction == 'right':
                        delta = width * 0.25
                        left += delta
                    elif direction == 'up':
                        delta = height * 0.25
                        top -= delta
                    elif direction == 'down':
                        delta = height * 0.25
                        top += delta

                    annotation_list[index]['height'] = min(max(height, 0.0), 100.0)
                    annotation_list[index]['width'] = min(max(width, 0.0), 100.0)
                    annotation_list[index]['left'] = min(max(left, 0.0), 100.0)
                    annotation_list[index]['top'] = min(max(top, 0.0), 100.0)

                    THROW_TEST_AOI_TURKING_MANIFEST.append(
                        {
                            'action': 'translation-%s' % (direction,),
                            'values': annotation_list[index],
                        }
                    )
            else:
                raise ValueError('Invalid throw_test_aoi_turking_mode')
            print(ut.repr3(THROW_TEST_AOI_TURKING_MANIFEST))

    THROW_TEST_AOI_TURKING_MANIFEST = ut.to_json(THROW_TEST_AOI_TURKING_MANIFEST)

    species_rowids = ibs._get_all_species_rowids()
    species_nice_list = ibs.get_species_nice(species_rowids)

    combined_list = sorted(zip(species_nice_list, species_rowids))
    species_nice_list = [combined[0] for combined in combined_list]
    species_rowids = [combined[1] for combined in combined_list]

    species_text_list = ibs.get_species_texts(species_rowids)
    species_list = list(zip(species_nice_list, species_text_list))
    species_list = [('Unspecified', const.UNKNOWN)] + species_list

    # Collect mapping of species to parts
    part_rowid_list = ibs.get_valid_part_rowids()
    part_aid_list = ibs.get_part_aids(part_rowid_list)
    part_species_text_list = ibs.get_annot_species_texts(part_aid_list)
    part_types_list = ibs.get_part_types(part_rowid_list)
    zipped = list(zip(part_species_text_list, part_types_list))

    species_part_dict = {const.UNKNOWN: set([])}
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

    if is_canonical:
        settings['ia-detection-setting-parts-show'] = True

    callback_url = '%s?imgsetid=%s' % (url_for('submit_detection'), imgsetid,)
    return appf.template(
        'turk',
        'detection',
        imgsetid=imgsetid,
        gid=gid,
        only_aid=only_aid,
        config_str=config_str,
        config=config,
        refer_aid=refer_aid,
        species=species,
        image_src=image_src,
        previous=previous,
        previous_only_aid=previous_only_aid,
        imagesettext=imagesettext,
        imagesettext_list_str=imagesettext_list_str,
        original_filename=original_filename,
        progress=progress,
        staged_progress=staged_progress,
        finished=finished,
        species_list=species_list,
        species_part_dict_json=species_part_dict_json,
        annotation_list=annotation_list,
        part_list=part_list,
        display_instructions=display_instructions,
        display_new_features=display_new_features,
        display_species_examples=display_species_examples,
        settings=settings,
        THROW_TEST_AOI_TURKING_AVAILABLE=THROW_TEST_AOI_TURKING_AVAILABLE,
        THROW_TEST_AOI_TURKING_MANIFEST=THROW_TEST_AOI_TURKING_MANIFEST,
        is_staged=is_staged,
        is_canonical=is_canonical,
        num_staged_aids=num_staged_aids,
        num_staged_part_rowids=num_staged_part_rowids,
        num_staged_sessions=num_staged_sessions,
        num_staged_users=num_staged_users,
        callback_url=callback_url,
        callback_method='POST',
        EMBEDDED=True,
        EMBEDDED_CSS=None,
        EMBEDDED_JAVASCRIPT=None,
        review=review,
    )


@register_route('/turk/detection/canonical/', methods=['GET'])
def turk_detection_canonical(
    aid=None, imgsetid=None, previous=None, previous_only_aid=None, **kwargs
):
    ibs = current_app.ibs

    imgsetid = None if imgsetid == '' or imgsetid == 'None' else imgsetid
    gid_list = ibs.get_valid_gids(imgsetid=imgsetid)
    aid_list = ut.flatten(ibs.get_image_aids(gid_list))
    aid_list = ibs.filter_annotation_set(aid_list, is_canonical=True)

    reviewed_list = appf.imageset_annot_canonical(ibs, aid_list)
    try:
        progress = '%0.2f' % (100.0 * reviewed_list.count(True) / len(aid_list),)
    except ZeroDivisionError:
        progress = '100.0'

    if aid is None:
        aid_list_ = ut.filterfalse_items(aid_list, reviewed_list)
        if len(aid_list_) == 0:
            aid = None
        else:
            aid = random.choice(aid_list_)

    gid = None
    finished = aid is None

    if not finished:
        gid = ibs.get_annot_gids(aid)

    args = (
        imgsetid,
        gid,
        aid,
        progress,
        previous,
        previous_only_aid,
    )
    print(
        'CANONICAL IMAGESETID: %s GID: %s AID: %s (PROG = %s, PREV GID = %s, PREV AID = %s)'
        % args
    )

    kwargs['canonical'] = True
    return turk_detection(
        gid,
        only_aid=aid,
        imgsetid=imgsetid,
        progress=progress,
        previous=previous,
        previous_only_aid=previous_only_aid,
        **kwargs,
    )


@register_route('/turk/detection/dynamic/', methods=['GET'])
def turk_detection_dynamic(**kwargs):
    ibs = current_app.ibs
    gid = request.args.get('gid', None)

    image_src = routes_ajax.image_src(gid)
    # Get annotations
    width, height = ibs.get_image_sizes(gid)
    aid_list = ibs.get_image_aids(gid)
    annot_bbox_list = ibs.get_annot_bboxes(aid_list)
    annot_thetas_list = ibs.get_annot_thetas(aid_list)
    species_list = ibs.get_annot_species_texts(aid_list)
    # Get annotation bounding boxes
    annotation_list = []
    for aid, annot_bbox, annot_theta, species in list(
        zip(aid_list, annot_bbox_list, annot_thetas_list, species_list)
    ):
        temp = {}
        temp['left'] = 100.0 * (annot_bbox[0] / width)
        temp['top'] = 100.0 * (annot_bbox[1] / height)
        temp['width'] = 100.0 * (annot_bbox[2] / width)
        temp['height'] = 100.0 * (annot_bbox[3] / height)
        temp['label'] = species
        temp['id'] = aid
        temp['theta'] = float(annot_theta)
        annotation_list.append(temp)
    if len(species_list) > 0:
        species = max(
            set(species_list), key=species_list.count
        )  # Get most common species
    elif appf.default_species(ibs) is not None:
        species = appf.default_species(ibs)
    else:
        species = KEY_DEFAULTS[SPECIES_KEY]

    callback_url = '%s?imgsetid=%s' % (url_for('submit_detection'), gid,)
    return appf.template(
        'turk',
        'detection_dynamic',
        gid=gid,
        refer_aid=None,
        species=species,
        image_src=image_src,
        annotation_list=annotation_list,
        callback_url=callback_url,
        callback_method='POST',
        EMBEDDED_CSS=None,
        EMBEDDED_JAVASCRIPT=None,
        __wrapper__=False,
    )


@register_route('/turk/annotation/', methods=['GET'])
def turk_annotation(**kwargs):
    """
    CommandLine:
        python -m wbia.web.app --exec-turk_annotation --db PZ_Master1

    Example:
        >>> # SCRIPT
        >>> from wbia.other.ibsfuncs import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb(defaultdb='PZ_Master1')
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
        gid = ibs.get_annot_gids(aid)
        image_src = routes_ajax.annotation_src(aid)
        species = ibs.get_annot_species_texts(aid)
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
        except Exception:
            pass
        gid = None
        image_src = None
        species = None
        viewpoint_value = -1
        quality_value = -1
        multiple_value = False

    imagesettext = ibs.get_imageset_text(imgsetid)

    species_rowids = ibs._get_all_species_rowids()
    species_nice_list = ibs.get_species_nice(species_rowids)

    combined_list = sorted(zip(species_nice_list, species_rowids))
    species_nice_list = [combined[0] for combined in combined_list]
    species_rowids = [combined[1] for combined in combined_list]

    species_text_list = ibs.get_species_texts(species_rowids)
    species_selected_list = [species == species_ for species_ in species_text_list]
    species_list = list(zip(species_nice_list, species_text_list, species_selected_list))
    species_list = [('Unspecified', const.UNKNOWN, True)] + species_list

    callback_url = url_for('submit_annotation')
    return appf.template(
        'turk',
        'annotation',
        imgsetid=imgsetid,
        src_ag=src_ag,
        dst_ag=dst_ag,
        gid=gid,
        aid=aid,
        viewpoint_value=viewpoint_value,
        quality_value=quality_value,
        multiple_value=multiple_value,
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
        review=review,
    )


@register_route('/turk/annotation/dynamic/', methods=['GET'])
def turk_annotation_dynamic(**kwargs):
    ibs = current_app.ibs
    aid = request.args.get('aid', None)
    imgsetid = request.args.get('imgsetid', None)

    review = 'review' in request.args.keys()
    gid = ibs.get_annot_gids(aid)
    image_src = routes_ajax.annotation_src(aid)
    species = ibs.get_annot_species_texts(aid)
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
    species_nice_list = [combined[0] for combined in combined_list]
    species_rowids = [combined[1] for combined in combined_list]

    species_text_list = ibs.get_species_texts(species_rowids)
    species_selected_list = [species == species_ for species_ in species_text_list]
    species_list = list(zip(species_nice_list, species_text_list, species_selected_list))
    species_list = [('Unspecified', const.UNKNOWN, True)] + species_list

    callback_url = url_for('submit_annotation')
    return appf.template(
        'turk',
        'annotation_dynamic',
        imgsetid=imgsetid,
        gid=gid,
        aid=aid,
        viewpoint_value=viewpoint_value,
        quality_value=quality_value,
        image_src=image_src,
        species_list=species_list,
        callback_url=callback_url,
        callback_method='POST',
        EMBEDDED_CSS=None,
        EMBEDDED_JAVASCRIPT=None,
        review=review,
        __wrapper__=False,
    )


@register_route('/turk/annotation/canonical/', methods=['GET'])
def turk_annotation_canonical(
    imgsetid=None, samples=200, species=None, version=1, **kwargs
):
    import random

    ibs = current_app.ibs

    if imgsetid is None:
        aid_list = ibs.get_valid_aids()
    else:
        aid_list = ibs.get_imageset_aids(imgsetid)

    if species is not None:
        aid_list = ibs.filter_annotation_set(aid_list, species=species)

    # enable_canonical = version == 1
    # aid_list = ibs.check_ggr_valid_aids(aid_list, species=species, threshold=0.75, enable_canonical=enable_canonical)

    # metadata_list = ibs.get_annot_metadata(aid_list)
    # canonical_flag_list = []
    # for metadata in metadata_list:
    #     turk = metadata.get('turk', {})
    #     canonical = turk.get('canonical', turk.get('grid', None))
    #     canonical_flag_list.append(canonical)

    canonical_flag_list = ibs.get_annot_canonical(aid_list)

    canonical_str = None
    if species == 'zebra_grevys':
        canonical_str = (
            "Grevy's Zebra - right side + shoulder chevron, side stripes, and hip chevron"
        )
    elif species == 'zebra_plains':
        canonical_str = (
            'Plains Zebra - left side + shoulder chevron, side stripes, and entire hip'
        )
    elif species == 'giraffe_reticulated':
        canonical_str = 'Reticulated Giraffe - left side + entire body center mass, neck'
    elif species == 'giraffe_masai':
        canonical_str = 'Masai Giraffe - right side + entire body center mass, neck'
    elif species == 'turtle_sea':
        canonical_str = 'Sea Turtle - right size + entire side of head'
    elif species == 'whale_fluke':
        canonical_str = (
            'Whale Fluke - top or bottom + trailing edge entirely out of water'
        )

    reviewed_list = []
    for canonical_flag in canonical_flag_list:
        if version in [1, 'set']:
            # Version 1 - Annotations that are unreviewed
            reviewed = canonical_flag in [True, False]
            version = 1
        elif version in [2, 'yes']:
            # Version 2 - Annotations that are marked YES as CA
            reviewed = canonical_flag in [None, False]
            version = 2
        elif version in [3, 'no']:
            # Version 2 - Annotations that are marked NO as CA
            reviewed = canonical_flag in [None, True]
            version = 3
        reviewed_list.append(reviewed)

    try:
        print('Total len(reviewed_list) = %d' % (len(reviewed_list),))
        progress = '%0.2f' % (100.0 * reviewed_list.count(True) / len(reviewed_list),)
    except ZeroDivisionError:
        progress = '100.0'

    COMPARE_TO_AOI = False
    if COMPARE_TO_AOI:
        kwargs = {
            'aoi_two_weight_filepath': 'ggr2',
        }
        prediction_list = ibs.depc_annot.get_property(
            'aoi_two', aid_list, 'class', config=kwargs
        )
        confidence_list = ibs.depc_annot.get_property(
            'aoi_two', aid_list, 'score', config=kwargs
        )
        confidence_list = [
            confidence if prediction == 'positive' else 1.0 - confidence
            for prediction, confidence in zip(prediction_list, confidence_list)
        ]
    else:
        confidence_list = [1.0] * len(aid_list)

    zipped = list(zip(aid_list, reviewed_list, confidence_list))
    values_list = ut.filterfalse_items(zipped, reviewed_list)

    aid_list_ = []
    highlighted_list_ = []
    confidence_list_ = []
    while len(values_list) > 0 and len(aid_list_) < samples:
        index = random.randint(0, len(values_list) - 1)
        aid, highlighted, confidence = values_list.pop(index)

        if version == 2:
            highlighted = True

        aid_list_.append(aid)
        highlighted_list_.append(highlighted)
        confidence_list_.append(confidence)

    finished = len(aid_list_) == 0

    annotation_list = list(zip(aid_list_, highlighted_list_, confidence_list_,))
    aid_list_str = ','.join(map(str, aid_list_))

    annotation_list.sort(key=lambda t: t[0])
    args = (
        url_for('submit_annotation_canonical'),
        imgsetid,
        version,
        samples,
        species,
    )
    callback_url = '%s?imgsetid=%s&version=%d&samples=%d&species=%s' % args
    return appf.template(
        'turk',
        'canonical',
        imgsetid=imgsetid,
        canonical_str=canonical_str,
        aid_list=aid_list_,
        aid_list_str=aid_list_str,
        num_aids=len(aid_list_),
        annotation_list=annotation_list,
        num_annotations=len(annotation_list),
        progress=progress,
        finished=finished,
        callback_url=callback_url,
        callback_method='POST',
    )


@register_route('/turk/splits/', methods=['GET'])
def turk_splits(aid=None, **kwargs):
    ibs = current_app.ibs

    annotation_list = []
    if aid is not None:
        nid = ibs.get_annot_nids(aid)
        aid_list = ibs.get_name_aids(nid)

        annotation_list = list(zip(aid_list,))

    annotation_list.sort(key=lambda t: t[0])
    callback_url = '%s' % (url_for('submit_splits'),)
    return appf.template(
        'turk',
        'splits',
        aid=aid,
        annotation_list=annotation_list,
        num_annotations=len(annotation_list),
        callback_url=callback_url,
        callback_method='POST',
    )


@register_route('/turk/part/contour/', methods=['GET'])
def turk_contour(part_rowid=None, imgsetid=None, previous=None, **kwargs):
    ibs = current_app.ibs

    default_list = [
        ('temp', True),
    ]

    config_kwargs = kwargs.get('config', {})
    config = {
        key: kwargs.get(key, config_kwargs.get(key, default))
        for key, default in default_list
    }
    config_str_list = [
        '%s=%s' % (key, 'true' if config[key] else 'false',) for key in config.keys()
    ]
    config_str = '&'.join(config_str_list)

    imgsetid = None if imgsetid == '' or imgsetid == 'None' else imgsetid
    gid_list = ibs.get_valid_gids(imgsetid=imgsetid)
    aid_list = ut.flatten(ibs.get_image_aids(gid_list))
    part_rowid_list = ut.flatten(ibs.get_annot_part_rowids(aid_list))
    part_rowid_list = list(set(part_rowid_list))

    reviewed_list = appf.imageset_part_contour_processed(ibs, part_rowid_list)

    try:
        progress = '%0.2f' % (100.0 * reviewed_list.count(True) / len(reviewed_list),)
    except ZeroDivisionError:
        progress = '100.0'

    imagesettext = None if imgsetid is None else ibs.get_imageset_text(imgsetid)

    if part_rowid is None:
        part_rowid_list_ = ut.filterfalse_items(part_rowid_list, reviewed_list)
        if len(part_rowid_list_) == 0:
            part_rowid = None
        else:
            part_rowid = random.choice(part_rowid_list_)

    finished = part_rowid is None
    display_instructions = request.cookies.get('ia-contour_instructions_seen', 1) == 1

    padding = 0.15
    if not finished:
        image_src = routes_ajax.part_src(part_rowid, pad=padding)

        # Get contours from part
        existing_contour_dict = ibs.get_part_contour(part_rowid)
        existing_contour = existing_contour_dict.get('contour', None)
        if existing_contour is None:
            existing_contour = {}
    else:
        image_src = None
        existing_contour = {}

    existing_contour_json = ut.to_json(existing_contour)

    settings_key_list = [
        ('ia-contour-setting-guiderail', '0'),
    ]

    settings = {
        settings_key: request.cookies.get(settings_key, settings_default) == '1'
        for (settings_key, settings_default) in settings_key_list
    }

    callback_url = '%s?imgsetid=%s' % (url_for('submit_contour'), imgsetid,)
    return appf.template(
        'turk',
        'contour',
        imgsetid=imgsetid,
        part_rowid=part_rowid,
        config_str=config_str,
        config=config,
        image_src=image_src,
        padding=padding,
        previous=previous,
        imagesettext=imagesettext,
        progress=progress,
        finished=finished,
        settings=settings,
        display_instructions=display_instructions,
        existing_contour_json=existing_contour_json,
        callback_url=callback_url,
        callback_method='POST',
    )


@register_route('/turk/species/', methods=['GET'])
def turk_species(hotkeys=8, refresh=False, previous_species_rowids=None, **kwargs):
    ibs = current_app.ibs
    tup = appf.get_turk_annot_args(appf.imageset_annot_processed)
    (aid_list, reviewed_list, imgsetid, src_ag, dst_ag, progress, aid, previous) = tup

    review = 'review' in request.args.keys()
    finished = aid is None
    display_instructions = request.cookies.get('ia-species_instructions_seen', 1) == 0
    if not finished:
        gid = ibs.get_annot_gids(aid)
        image_src = routes_ajax.annotation_src(aid, resize=None)
        species = ibs.get_annot_species_texts(aid)
    else:
        try:
            ibs.update_special_imagesets()
            ibs.notify_observers()
        except Exception:
            pass
        gid = None
        image_src = None
        species = None

    imagesettext = ibs.get_imageset_text(imgsetid)
    species_rowids = ibs._get_all_species_rowids()

    if previous_species_rowids is not None:
        try:
            for previous_species_rowid in previous_species_rowids:
                assert previous_species_rowid in species_rowids

            species_rowids = previous_species_rowids
        except Exception:
            print('Error finding previous species rowid in existing list')
            previous_species_rowids = None

    if previous_species_rowids is None:
        refresh = True

    current_species_rowids = ibs.get_annot_species_rowids(aid_list)

    species_count = [
        len(current_species_rowids) - current_species_rowids.count(species_rowid)
        for species_rowid in species_rowids
    ]
    species_nice_list = ibs.get_species_nice(species_rowids)
    combined_list = list(zip(species_count, species_nice_list, species_rowids))

    if refresh:
        print('REFRESHING!')
        combined_list = sorted(combined_list)

    species_count_list = [combined[0] for combined in combined_list]
    species_nice_list = [combined[1] for combined in combined_list]
    species_rowids = [combined[2] for combined in combined_list]
    species_text_list = ibs.get_species_texts(species_rowids)

    species_rowids_json = ut.to_json(species_rowids)

    hotkey_list = [index + 1 for index in range(len(species_nice_list))]
    species_selected_list = [species == species_ for species_ in species_text_list]
    species_list = list(
        zip(hotkey_list, species_nice_list, species_text_list, species_selected_list)
    )
    species_extended_list = []
    other_selected = False

    zipped = list(zip(species_count_list, species_nice_list, species_selected_list))
    for index, (species_count, species_nice, species_selected) in enumerate(zipped):
        if species_selected:
            species_nice += ' (default)'
        args = (
            len(current_species_rowids) - species_count,
            species_nice,
        )
        if index >= hotkeys:
            print('% 5d   : %s' % args)
        else:
            print('% 5d * : %s' % args)

    if len(species_list) >= hotkeys:
        species_extended_list = species_list[hotkeys:]
        species_list = species_list[:hotkeys]
        extended_flag_list = [_[3] for _ in species_extended_list]

        if True in extended_flag_list:
            other_selected = True

    species_list = species_list + [
        (len(species_list) + 1, 'Other', const.UNKNOWN, other_selected)
    ]

    example_species_list = [
        (key, const.SPECIES_MAPPING[key][1])
        for key in sorted(list(const.SPECIES_MAPPING.keys()))
        if const.SPECIES_MAPPING[key][0] is not None and key != const.UNKNOWN
    ]

    callback_url = url_for('submit_species')
    return appf.template(
        'turk',
        'species',
        imgsetid=imgsetid,
        src_ag=src_ag,
        dst_ag=dst_ag,
        gid=gid,
        aid=aid,
        image_src=image_src,
        previous=previous,
        species_list=species_list,
        species_extended_list=species_extended_list,
        species_rowids_json=species_rowids_json,
        example_species_list=example_species_list,
        imagesettext=imagesettext,
        progress=progress,
        finished=finished,
        display_instructions=display_instructions,
        callback_url=callback_url,
        callback_method='POST',
        EMBEDDED_CSS=None,
        EMBEDDED_JAVASCRIPT=None,
        review=review,
    )


@register_route('/turk/part/type/', methods=['GET'])
def turk_part_types(
    part_rowid=None,
    imgsetid=None,
    previous=None,
    hotkeys=8,
    refresh=False,
    previous_part_types=None,
    **kwargs,
):
    ibs = current_app.ibs

    imgsetid = None if imgsetid == '' or imgsetid == 'None' else imgsetid
    gid_list = ibs.get_valid_gids(imgsetid=imgsetid)
    aid_list = ut.flatten(ibs.get_image_aids(gid_list))
    part_rowid_list = ut.flatten(ibs.get_annot_part_rowids(aid_list))
    part_rowid_list = list(set(part_rowid_list))

    reviewed_list = appf.imageset_part_type_processed(ibs, part_rowid_list)

    try:
        progress = '%0.2f' % (100.0 * reviewed_list.count(True) / len(part_rowid_list),)
    except ZeroDivisionError:
        progress = '0.00'
    part_rowid = request.args.get('part_rowid', '')
    if len(part_rowid) > 0:
        part_rowid = int(part_rowid)
    else:
        part_rowid_list_ = ut.filterfalse_items(part_rowid_list, reviewed_list)
        if len(part_rowid_list_) == 0:
            part_rowid = None
        else:
            part_rowid = random.choice(part_rowid_list_)

    padding = 0.15
    review = 'review' in request.args.keys()
    finished = part_rowid is None
    display_instructions = request.cookies.get('ia-part-type_instructions_seen', 1) == 0
    if not finished:
        aid = ibs.get_part_aids(part_rowid)
        gid = ibs.get_annot_gids(aid)
        image_src = routes_ajax.part_src(part_rowid, pad=padding)
        part_type = ibs.get_part_types(part_rowid)
    else:
        aid = None
        gid = None
        image_src = None
        part_type = None

    imagesettext = ibs.get_imageset_text(imgsetid)

    all_part_rowid_list = ibs._get_all_part_rowids()
    all_part_types = sorted(set(ibs.get_part_types(all_part_rowid_list)))

    if ibs.containerized:
        hostname = const.CONTAINER_NAME
    else:
        hostname = ibs.dbname

    if ibs.dbname == 'WD_Master' or hostname == 'wilddog':
        all_part_types = all_part_types + [
            'standard',
            'short_black',
            'long_black',
            'double_black_brown',
            'double_black_white',
            'triple_black',
            'long_white',
        ]
        all_part_types = sorted(list(set(all_part_types) - set([const.UNKNOWN])))

    if not refresh and previous_part_types is not None:
        try:
            for previous_part_type in previous_part_types:
                assert previous_part_type in all_part_types

            all_part_types = previous_part_types
        except Exception:
            print('Error finding previous part_type in existing list')
            previous_part_types = None

    if previous_part_types is None:
        refresh = True

    current_part_types = ibs.get_part_types(all_part_rowid_list)

    all_part_type_count = [
        len(current_part_types) - current_part_types.count(all_part_type)
        for all_part_type in all_part_types
    ]
    all_part_nices = [
        const.PARTS_MAPPING.get(all_part_type, all_part_type)
        for all_part_type in all_part_types
    ]
    combined_list = list(zip(all_part_type_count, all_part_nices, all_part_types))

    if refresh:
        print('REFRESHING!')
        combined_list = sorted(combined_list)

    part_type_count_list = [combined[0] for combined in combined_list]
    part_nice_list = [combined[1] for combined in combined_list]
    part_type_list = [combined[2] for combined in combined_list]
    part_type_list_json = ut.to_json(part_type_list)

    hotkey_list = [index + 1 for index in range(len(part_type_list))]
    part_type_selected_list = [part_type == part_type_ for part_type_ in part_type_list]
    part_type_option_list = list(
        zip(hotkey_list, part_nice_list, part_type_list, part_type_selected_list)
    )
    part_type_extended_list = []
    other_selected = False

    zipped = list(zip(part_type_count_list, part_nice_list, part_type_selected_list))
    for index, (part_type_count, part_nice, part_type_selected) in enumerate(zipped):
        if part_type_selected:
            part_nice += ' (default)'
        args = (
            len(current_part_types) - part_type_count,
            part_nice,
        )
        if index >= hotkeys:
            print('% 5d   : %s' % args)
        else:
            print('% 5d * : %s' % args)

    if len(part_type_option_list) >= hotkeys:
        part_type_extended_list = part_type_option_list[hotkeys:]
        part_type_option_list = part_type_option_list[:hotkeys]
        extended_flag_list = [_[3] for _ in part_type_extended_list]

        if True in extended_flag_list:
            other_selected = True

    part_type_option_list = part_type_option_list + [
        (len(part_type_option_list) + 1, 'Other', const.UNKNOWN, other_selected)
    ]

    callback_url = url_for('submit_part_types')
    return appf.template(
        'turk',
        'part_type',
        imgsetid=imgsetid,
        part_rowid=part_rowid,
        gid=gid,
        aid=aid,
        image_src=image_src,
        previous=previous,
        part_type_option_list=part_type_option_list,
        part_type_extended_list=part_type_extended_list,
        part_type_list_json=part_type_list_json,
        imagesettext=imagesettext,
        progress=progress,
        finished=finished,
        display_instructions=display_instructions,
        callback_url=callback_url,
        callback_method='POST',
        EMBEDDED_CSS=None,
        EMBEDDED_JAVASCRIPT=None,
        review=review,
    )


@register_route('/turk/viewpoint/', methods=['GET'])
def turk_viewpoint(**kwargs):
    """
    CommandLine:
        python -m wbia.web.app --exec-turk_viewpoint --db PZ_Master1

    Example:
        >>> # SCRIPT
        >>> from wbia.other.ibsfuncs import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb(defaultdb='PZ_Master1')
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
        gid = ibs.get_annot_gids(aid)
        image_src = routes_ajax.annotation_src(aid)
        species = ibs.get_annot_species_texts(aid)
    else:
        gid = None
        image_src = None
        species = None

    imagesettext = ibs.get_imageset_text(imgsetid)

    species_rowids = ibs._get_all_species_rowids()
    species_nice_list = ibs.get_species_nice(species_rowids)

    combined_list = sorted(zip(species_nice_list, species_rowids))
    species_nice_list = [combined[0] for combined in combined_list]
    species_rowids = [combined[1] for combined in combined_list]

    species_text_list = ibs.get_species_texts(species_rowids)
    species_selected_list = [species == species_ for species_ in species_text_list]
    species_list = list(zip(species_nice_list, species_text_list, species_selected_list))
    species_list = [('Unspecified', const.UNKNOWN, True)] + species_list

    return appf.template(
        'turk',
        'viewpoint',
        imgsetid=imgsetid,
        src_ag=src_ag,
        dst_ag=dst_ag,
        gid=gid,
        aid=aid,
        value=value,
        image_src=image_src,
        previous=previous,
        species_list=species_list,
        imagesettext=imagesettext,
        progress=progress,
        finished=finished,
        display_instructions=display_instructions,
        review=review,
    )


@register_route('/turk/viewpoint2/', methods=['GET'])
def turk_viewpoint2(**kwargs):
    """
    CommandLine:
        python -m wbia.web.app --exec-turk_viewpoint --db PZ_Master1

    Example:
        >>> # SCRIPT
        >>> from wbia.other.ibsfuncs import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb(defaultdb='PZ_Master1')
        >>> aid_list_ = ibs.find_unlabeled_name_members(suspect_yaws=True)
        >>> aid_list = ibs.filter_aids_to_quality(aid_list_, 'good', unknown_ok=False)
        >>> ibs.start_web_annot_groupreview(aid_list)
    """
    ibs = current_app.ibs
    tup = appf.get_turk_annot_args(appf.imageset_annot_viewpoint_processed)
    (aid_list, reviewed_list, imgsetid, src_ag, dst_ag, progress, aid, previous) = tup

    viewpoint = ibs.get_annot_viewpoints(aid)
    viewpoint1, viewpoint2, viewpoint3 = appf.convert_viewpoint_to_tuple(viewpoint)

    review = 'review' in request.args.keys()
    finished = aid is None
    display_instructions = request.cookies.get('ia-viewpoint_instructions_seen', 1) == 0
    if not finished:
        gid = ibs.get_annot_gids(aid)
        image_src = routes_ajax.annotation_src(aid)
        species = ibs.get_annot_species_texts(aid)
    else:
        gid = None
        image_src = None
        species = None

    imagesettext = ibs.get_imageset_text(imgsetid)

    species_rowids = ibs._get_all_species_rowids()
    species_nice_list = ibs.get_species_nice(species_rowids)

    combined_list = sorted(zip(species_nice_list, species_rowids))
    species_nice_list = [combined[0] for combined in combined_list]
    species_rowids = [combined[1] for combined in combined_list]

    species_text_list = ibs.get_species_texts(species_rowids)
    species_selected_list = [species == species_ for species_ in species_text_list]
    species_list = list(zip(species_nice_list, species_text_list, species_selected_list))
    species_list = [('Unspecified', const.UNKNOWN, True)] + species_list

    return appf.template(
        'turk',
        'viewpoint2',
        imgsetid=imgsetid,
        src_ag=src_ag,
        dst_ag=dst_ag,
        gid=gid,
        aid=aid,
        viewpoint1=viewpoint1,
        viewpoint2=viewpoint2,
        viewpoint3=viewpoint3,
        image_src=image_src,
        previous=previous,
        species_list=species_list,
        imagesettext=imagesettext,
        progress=progress,
        finished=finished,
        display_instructions=display_instructions,
        review=review,
    )


@register_route('/turk/viewpoint3/', methods=['GET'])
def turk_viewpoint3(**kwargs):
    with ut.Timer('[turk_viewpoint3]'):
        ibs = current_app.ibs
        tup = appf.get_turk_annot_args(appf.imageset_annot_viewpoint_processed)
        (
            aid_list,
            reviewed_list,
            imgsetid,
            src_ag,
            dst_ag,
            progress,
            aid,
            previous,
        ) = tup

        viewpoint = None if aid is None else ibs.get_annot_viewpoints(aid)
        viewpoint_code = const.YAWALIAS.get(viewpoint, None)

        review = 'review' in request.args.keys()
        finished = aid is None
        display_instructions = (
            request.cookies.get('ia-viewpoint3_instructions_seen', 1) == 0
        )
        if not finished:
            gid = ibs.get_annot_gids(aid)
            image_src = routes_ajax.annotation_src(aid)
            species = ibs.get_annot_species_texts(aid)
        else:
            gid = None
            image_src = None
            species = None

        imagesettext = ibs.get_imageset_text(imgsetid)

        species_rowids = ibs._get_all_species_rowids()
        species_nice_list = ibs.get_species_nice(species_rowids)

        combined_list = sorted(zip(species_nice_list, species_rowids))
        species_nice_list = [combined[0] for combined in combined_list]
        species_rowids = [combined[1] for combined in combined_list]

        species_text_list = ibs.get_species_texts(species_rowids)
        species_selected_list = [species == species_ for species_ in species_text_list]
        species_list = list(
            zip(species_nice_list, species_text_list, species_selected_list)
        )
        species_list = [('Unspecified', const.UNKNOWN, True)] + species_list

        axis_preference = request.cookies.get('ia-viewpoint3_axis_preference', None)

        template = appf.template(
            'turk',
            'viewpoint3',
            imgsetid=imgsetid,
            src_ag=src_ag,
            dst_ag=dst_ag,
            gid=gid,
            aid=aid,
            viewpoint_code=viewpoint_code,
            axis_preference=axis_preference,
            image_src=image_src,
            previous=previous,
            species_list=species_list,
            imagesettext=imagesettext,
            progress=progress,
            finished=finished,
            display_instructions=display_instructions,
            review=review,
        )
    return template


def commit_current_query_object_names(query_object, ibs):
    r"""
    Args:
        query_object (wbia.AnnotInference):
        ibs (wbia.IBEISController):  image analysis api
    """
    # Ensure connected components are used to relabel names
    query_object.relabel_using_reviews()
    # Transfers any remaining internal feedback into staging
    # TODO:  uncomment once buffer is dead
    # query_object.write_wbia_staging_feedback()
    # Commit a delta of the current annotmatch
    query_object.write_wbia_annotmatch_feedback()
    query_object.write_wbia_name_assignment()


def precompute_current_review_match_images(
    ibs,
    query_object,
    global_feedback_limit=GLOBAL_FEEDBACK_LIMIT,
    view_orientation='vertical',
):
    from wbia.web import apis_query

    review_aid1_list, review_aid2_list = query_object.get_filtered_edges(
        GLOBAL_FEEDBACK_CONFIG_DICT
    )
    qreq_ = query_object.qreq_

    assert len(review_aid1_list) == len(review_aid2_list), 'not aligned'

    # Precompute
    zipped = list(zip(review_aid1_list, review_aid2_list))
    prog = ut.ProgIter(
        enumerate(zipped), length=len(review_aid2_list), label='Rending images'
    )
    for index, (aid1, aid2) in prog:
        if index > global_feedback_limit * 2:
            break
        cm, aid1, aid2 = query_object.lookup_cm(aid1, aid2)
        try:
            apis_query.ensure_review_image(
                ibs, aid2, cm, qreq_, view_orientation=view_orientation
            )
        except KeyError as ex:
            ut.printex(
                ex,
                'Failed to make review image. falling back',
                tb=True,
                keys=['cm.qaid', 'aid2'],
                iswarning=True,
            )
            apis_query.ensure_review_image(
                ibs,
                aid2,
                cm,
                qreq_,
                view_orientation=view_orientation,
                draw_matches=False,
            )


@register_ibs_method
def load_identification_query_object_worker(ibs, **kwargs):
    _init_identification_query_object(ibs, **kwargs)
    return 'precomputed'


def _init_identification_query_object(
    ibs,
    debug_ignore_name_gt=False,
    global_feedback_limit=GLOBAL_FEEDBACK_LIMIT,
    **kwargs,
):
    """
    CommandLine:
        python -m wbia.web.routes _init_identification_query_object

    Ignore:
        # mount lev to the home drive
        sshfs -o idmap=user lev:/ ~/lev

        # unmount
        fusermount -u ~/lev

        import wbia
        ibs = wbia.opendb(ut.truepath('~/lev/media/hdd/work/EWT_Cheetahs'))
        aid_list = ibs.filter_annots_general(view=['right', 'frontright', 'backright'])
        infr = wbia.AnnotInference(ibs, aid_list, autoinit=True)
        infr.reset_feedback('staging')
        infr.apply_feedback_edges()
        infr.exec_matching()
        infr.apply_match_edges()
        infr.apply_match_scores()

    Example:
        >>> # SLOW_DOCTEST
        >>> from wbia.web.routes import *  # NOQA
        >>> from wbia.web.routes import _init_identification_query_object
        >>> import wbia
        >>> ibs = wbia.opendb('PZ_MTEST')
        >>> ut.exec_funckw(_init_identification_query_object, locals())
        >>> kwargs = {}
        >>> _init_identification_query_object(ibs)

    """
    import wbia

    if ibs.dbname == 'EWT_Cheetahs':
        aid_list = ibs.filter_annots_general(view=['right', 'frontright', 'backright'])
    else:
        aid_list = ibs.get_valid_aids(is_exemplar=True)

    nids = [-aid for aid in aid_list] if debug_ignore_name_gt else None

    # Initailize a graph with no edges.
    query_object = wbia.AnnotInference(ibs, aid_list, nids=nids, autoinit=True)

    # Load feedback from the staging database (does not change graph state)
    query_object.reset_feedback('staging', apply=True)

    query_object.refresh_candidate_edges()

    # Exec matching (adds candidate edges using hotspotter and score them)
    query_object.exec_matching()
    query_object.apply_match_edges()
    query_object.apply_match_scores()

    # Use connected components to relabel names on nodes
    query_object.relabel_using_reviews()

    # Create a priority on edge review ands determines inconsistencies
    query_object.ensure_mst()
    query_object.apply_nondynamic_update()

    print('Precomputing match images')
    # view_orientation = request.args.get('view_orientation', 'vertical')
    # precompute_current_review_match_images(ibs, query_object,
    #                                        global_feedback_limit=global_feedback_limit,
    #                                        view_orientation='vertical')
    # precompute_current_review_match_images(ibs, query_object,
    #                                        global_feedback_limit=global_feedback_limit,
    #                                        view_orientation='horizontal')
    return query_object


def load_identification_query_object(
    autoinit=False, global_feedback_limit=GLOBAL_FEEDBACK_LIMIT, **kwargs
):
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
        print('Popping %r out of QUERY_OBJECT_FEEDBACK_BUFFER' % (feedback,))
        aid1, aid2, state, tags = feedback
        query_object.add_feedback((aid1, aid2), state, tags=tags)
        query_object.GLOBAL_FEEDBACK_COUNTER += 1

    return query_object


def check_engine_identification_query_object(
    global_feedback_limit=GLOBAL_FEEDBACK_LIMIT,
):
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
        # import wbia
        # web_ibs = wbia.opendb_bg_web(dbdir=ibs.dbdir, port=6000)
        # query_object_jobid = web_ibs.send_wbia_request('/api/engine/query/graph/')
        # print('query_object_jobid = %r' % (query_object_jobid, ))
        # current_app.QUERY_OBJECT_JOBID = query_object_jobid

    query_object_status_dict = ibs.get_job_status(current_app.QUERY_OBJECT_JOBID)
    args = (
        current_app.QUERY_OBJECT_JOBID,
        query_object_status_dict,
    )
    print('job id %r: %r' % args)

    if query_object_status_dict['jobstatus'] == 'completed':
        query_object_result = ibs.get_job_result(current_app.QUERY_OBJECT_JOBID)
        args = (
            current_app.QUERY_OBJECT_JOBID,
            query_object_result,
        )
        print('job id %r: %r' % args)
        assert query_object_result['json_result'] == 'precomputed'
        return True

    return False


@register_route('/turk/identification/lnbnn/', methods=['GET'])
def turk_identification(
    aid1=None,
    aid2=None,
    use_engine=False,
    global_feedback_limit=GLOBAL_FEEDBACK_LIMIT,
    **kwargs,
):
    """
    CommandLine:
        python -m wbia.web.routes turk_identification --db PZ_Master1
        python -m wbia.web.routes turk_identification --db PZ_MTEST
        python -m wbia.web.routes turk_identification --db testdb1 --show

    Example:
        >>> # SCRIPT
        >>> from wbia.other.ibsfuncs import *  # NOQA
        >>> import wbia
        >>> with wbia.opendb_bg_web('testdb1', managed=True) as web_ibs:
        ...     resp = web_ibs.get('/turk/identification/lnbnn/')
        >>> ut.quit_if_noshow()
        >>> import wbia.plottool as pt
        >>> ut.render_html(resp.content)
        >>> ut.show_if_requested()
    """
    from wbia.web import apis_query

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
                raw_review_list = []

                # Get actual
                review_cfg[
                    'max_num'
                ] = global_feedback_limit  # Controls the top X to be randomly sampled and displayed to all concurrent users
                values = query_object.pop()
                (review_aid1_list, review_aid2_list), review_confidence = values
                review_aid1_list = [review_aid1_list]
                review_aid2_list = [review_aid2_list]

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
                print(
                    'Feedback counter    = %r / %r'
                    % (query_object.GLOBAL_FEEDBACK_COUNTER, GLOBAL_FEEDBACK_LIMIT,)
                )
                print('Status dict         = %r' % (status_dict,))
                print('Raw list len        = %r' % (len(raw_review_list),))
                print('len(query_aid_list) = %r' % (len(query_object.aids),))
                print('Estimated remaining = %r' % (status_remaining,))
                print('Reviews list len    = %r' % (len(review_aid1_list),))
                try:
                    progress = '%0.02f' % (
                        100.0 * (1.0 - (status_remaining / len(query_object.aids))),
                    )
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
                            print('Picked random index = %r' % (index,))
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
                        view_orientation = request.args.get(
                            'view_orientation', 'vertical'
                        )
                        with ut.Timer(
                            '[web.routes.turk_identification] ... ... Render images2'
                        ):
                            try:
                                image_matches, _ = apis_query.ensure_review_image(
                                    ibs,
                                    aid2,
                                    cm,
                                    qreq_,
                                    view_orientation=view_orientation,
                                )
                            except KeyError as ex:
                                ut.printex(
                                    ex,
                                    'Failed to make review image',
                                    tb=True,
                                    keys=['cm.qaid', 'aid1', 'aid2'],
                                    iswarning=True,
                                )
                            try:
                                image_clean, _ = apis_query.ensure_review_image(
                                    ibs,
                                    aid2,
                                    cm,
                                    qreq_,
                                    view_orientation=view_orientation,
                                    draw_matches=False,
                                )
                            except KeyError as ex:
                                image_clean = np.zeros((100, 100, 3), dtype=np.uint8)
                                ut.printex(
                                    ex,
                                    'Failed to make fallback review image',
                                    tb=True,
                                    keys=['cm.qaid', 'aid1', 'aid2'],
                                )

                        with ut.Timer(
                            '[web.routes.turk_identification] ... ... Embed images'
                        ):
                            image_matches_src = appf.embed_image_html(image_matches)
                            image_clean_src = appf.embed_image_html(image_clean)

                    with ut.Timer(
                        '[web.routes.turk_identification] ... Process previous'
                    ):
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

    confidence_dict = const.CONFIDENCE.NICE_TO_CODE
    confidence_nice_list = confidence_dict.keys()
    confidence_text_list = confidence_dict.values()
    confidence_selected_list = [
        confidence_text == 'unspecified' for confidence_text in confidence_text_list
    ]
    confidence_list = list(
        zip(confidence_nice_list, confidence_text_list, confidence_selected_list)
    )

    timedelta_str = 'Unknown'
    if aid1 is not None and aid2 is not None:
        unixtime_list = ibs.get_image_unixtime(ibs.get_annot_gids([aid1, aid2]))
        if -1.0 not in unixtime_list and 0.0 not in unixtime_list:
            timedelta = abs(unixtime_list[1] - unixtime_list[0])
            secs = 60.0
            mins = 60.0 * 60.0
            hrs = 60.0 * 60.0 * 24.0
            days = 60.0 * 60.0 * 24.0 * 365.0

            if timedelta < secs:
                timedelta_str = '%0.2f seconds' % (timedelta,)
            elif timedelta < mins:
                timedelta /= secs
                timedelta_str = '%0.2f minutes' % (timedelta,)
            elif timedelta < hrs:
                timedelta /= mins
                timedelta_str = '%0.2f hours' % (timedelta,)
            elif timedelta < days:
                timedelta /= hrs
                timedelta_str = '%0.2f days' % (timedelta,)
            else:
                timedelta /= days
                timedelta_str = '%0.2f years' % (timedelta,)

    callback_url = url_for('submit_identification')
    return appf.template(
        'turk',
        'identification',
        match_data='Score: ' + str(match_score),
        image_clean_src=image_clean_src,
        image_matches_src=image_matches_src,
        aid1=aid1,
        aid2=aid2,
        progress=progress,
        session=True,
        session_counter=session_counter,
        session_limit=session_limit,
        confidence_list=confidence_list,
        timedelta_str=timedelta_str,
        finished=finished,
        annot_uuid_1=str(annot_uuid_1),
        annot_uuid_2=str(annot_uuid_2),
        previous=previous,
        previous_version=1,
        replace_review_rowid=replace_review_rowid,
        view_orientation=view_orientation,
        callback_url=callback_url,
        callback_method='POST',
        EMBEDDED_CSS=None,
        EMBEDDED_JAVASCRIPT=None,
    )


@register_ibs_method
def _princeton_kaia_annot_filtering(ibs, current_aids, desired_species):
    total = 0
    species_dict = {'zebra': []}
    species_list = ibs.get_annot_species_texts(current_aids)
    viewpoint_list = ibs.get_annot_viewpoints(current_aids)
    for aid, species, viewpoint in zip(current_aids, species_list, viewpoint_list):
        if viewpoint is None:
            # print(aid)
            continue
        if species not in species_dict:
            species_dict[species] = []
        if species == 'zebra_plains':
            if 'left' in viewpoint:
                species_dict[species].append(aid)
                species_dict['zebra'].append(aid)
                total += 1
        if species == 'zebra_grevys':
            if 'right' in viewpoint:
                species_dict[species].append(aid)
                species_dict['zebra'].append(aid)
                total += 1
        if species == 'zebra_hybrid':
            if 'right' in viewpoint:
                species_dict[species].append(aid)
                species_dict['zebra'].append(aid)
                total += 1

    aid_list = species_dict[desired_species]

    metadata_dict_list = ibs.get_annot_metadata(aid_list)
    excluded_list = [
        metadata_dict.get('excluded', False) for metadata_dict in metadata_dict_list
    ]
    new_aid_list = ut.compress(aid_list, ut.not_list(excluded_list))
    new_aid_list = list(set(new_aid_list))

    return new_aid_list


@register_ibs_method
def _princeton_kaia_imageset_filtering(ibs, year=2019, **kwargs):
    whitelist_list = [
        # 'Miscellaneous Found Images',
        # 'Candidate Images',
    ]
    include_list = [
        'Kaia Field Season',
        'Kaia Loops',
        'More Photo from Loops',
    ]
    exclude_list = [
        # ('August', '2015'),
        # ('September', '2015'),
        # ('October', '2015'),
        # ('December', '2015'),
        # ('February', '2016'),
        # ('March', '2016'),
        # ('April', '2016'),
        # ('May', '2016'),
        # ('June', '2016'),
        # ('July', '2016'),
    ]

    imageset_rowid_list = ibs.get_valid_imgsetids()
    imageset_text_list = ibs.get_imageset_text(imageset_rowid_list)

    years = []
    invalid_text_list = []
    valid_imageset_rowid_list = []
    for imageset_rowid, imageset_text in zip(imageset_rowid_list, imageset_text_list):
        if imageset_text in whitelist_list:
            valid_imageset_rowid_list.append(imageset_rowid)
            continue
        accepted_years = list(map(str, list(range(2015, year + 1))))
        flag = False
        for accepted_year in accepted_years:
            if accepted_year in imageset_text:
                years.append(accepted_year)
                flag = True
        if not flag:
            invalid_text_list.append(imageset_text)
            continue
        for include_ in include_list:
            if include_ in imageset_text:
                valid = True
                for exclude1, exclude2 in exclude_list:
                    if exclude1 in imageset_text and exclude2 in imageset_text:
                        valid = False
                        break
                if valid:
                    valid_imageset_rowid_list.append(imageset_rowid)
                else:
                    invalid_text_list.append(imageset_text)

    valid_imageset_rowid_list = list(set(valid_imageset_rowid_list))

    return valid_imageset_rowid_list


@register_ibs_method
def _princeton_kaia_filtering(
    ibs, current_aids=None, desired_species=None, tier=1, year=2019, **kwargs
):

    valid_imageset_rowid_list = ibs._princeton_kaia_imageset_filtering(year=year)

    if tier in [0]:
        candidate_aids_list = ibs.get_imageset_aids(valid_imageset_rowid_list)
        candidate_aid_list = list(set(ut.flatten(candidate_aids_list)))
        aids = ibs._princeton_kaia_annot_filtering(candidate_aid_list, desired_species)

        config = {
            'classifier_algo': 'densenet',
            'classifier_weight_filepath': 'canonical_zebra_grevys_v4',
        }
        prediction_list = ibs.depc_annot.get_property(
            'classifier', aids, 'class', config=config
        )
        confidence_list = ibs.depc_annot.get_property(
            'classifier', aids, 'score', config=config
        )
        confidence_list = [
            confidence if prediction == 'positive' else 1.0 - confidence
            for prediction, confidence in zip(prediction_list, confidence_list)
        ]
        flag_list = [confidence >= 0.31 for confidence in confidence_list]

        ca_aids = ut.compress(aids, flag_list)
        new_aid_list = sorted(list(set(ca_aids)))
        return new_aid_list

    if tier in [6]:
        all_aid_list = ibs.get_valid_aids()
        all_nid_list = ibs.get_annot_nids(all_aid_list)
        flag_list = [nid > 0 for nid in all_nid_list]
        new_aid_list = ut.compress(all_aid_list, flag_list)
        new_nid_list = ut.compress(all_nid_list, flag_list)

        gid_list = list(set(ibs.get_annot_gids(new_aid_list)))
        imageset_rowid_list = list(set(ut.flatten(ibs.get_image_imgsetids(gid_list))))
        imageset_rowid_list_ = list(
            set(imageset_rowid_list) & set(valid_imageset_rowid_list)
        )
        candidate_aid_list = list(
            set(ut.flatten(ibs.get_imageset_aids(imageset_rowid_list_)))
        )

        new_aid_list = ibs._princeton_kaia_annot_filtering(
            candidate_aid_list, desired_species
        )

    if current_aids is None:
        current_imageset_rowid = ibs.get_imageset_imgsetids_from_text('Candidate Images')
        current_gids = ibs.get_imageset_gids(current_imageset_rowid)
        current_aids = ut.flatten(ibs.get_image_aids(current_gids))
        current_aoi_list = ibs.get_annot_interest(current_aids)
        current_aids = ut.compress(current_aids, current_aoi_list)

    # current_nids = ibs.get_annot_nids(current_aids)

    # x = [current_nid for current_nid in current_nids if current_nid <= 0]
    # print(len(x))

    new_aid_list = ibs._princeton_kaia_annot_filtering(current_aids, desired_species)

    if tier in [2, 3, 4, 5]:
        # Get all siblings of these animals
        gid_list = list(set(ibs.get_annot_gids(new_aid_list)))
        current_aids = ut.flatten(ibs.get_image_aids(gid_list))
        new_aid_list = ibs._princeton_kaia_annot_filtering(current_aids, desired_species)

    if tier in [3]:
        new_gid_list = list(
            set(ut.flatten(ibs.get_imageset_gids(valid_imageset_rowid_list)))
        )
        new_aid_list_ = ut.flatten(ibs.get_image_aids(new_gid_list))
        new_aid_list_ = ibs._princeton_kaia_annot_filtering(
            new_aid_list_, desired_species
        )

        new_aid_list = new_aid_list + new_aid_list_
        new_aid_list = list(set(new_aid_list))

    if tier in [4, 5]:
        ibs.delete_empty_nids()

        valid_nid_set = set([])
        new_nid_list = ibs.get_annot_nids(new_aid_list)
        for new_aid, new_nid in zip(new_aid_list, new_nid_list):
            if new_nid > 0:
                valid_nid_set.add(new_nid)

        all_aid_list = ibs.get_valid_aids()
        all_aid_list = ibs._princeton_kaia_annot_filtering(all_aid_list, desired_species)
        all_nid_list = ibs.get_annot_nids(all_aid_list)
        all_flag_list = [all_nid in valid_nid_set for all_nid in all_nid_list]

        named_aid_list = ut.compress(all_aid_list, all_flag_list)
        new_aid_list = sorted(list(set(new_aid_list + named_aid_list)))

    if tier in [5]:
        gid_list = list(set(ibs.get_annot_gids(new_aid_list)))
        imageset_rowid_list = list(set(ut.flatten(ibs.get_image_imgsetids(gid_list))))
        imageset_rowid_list_ = list(
            set(imageset_rowid_list) & set(valid_imageset_rowid_list)
        )
        candidate_aid_list = list(
            set(ut.flatten(ibs.get_imageset_aids(imageset_rowid_list_)))
        )

        new_aid_list = ibs._princeton_kaia_annot_filtering(
            candidate_aid_list, desired_species
        )

    return new_aid_list


def _zebra_annot_filtering(ibs, current_aids, desired_species):
    total = 0
    species_dict = {}
    species_list = ibs.get_annot_species_texts(current_aids)
    viewpoint_list = ibs.get_annot_viewpoints(current_aids)
    for aid, species, viewpoint in zip(current_aids, species_list, viewpoint_list):
        if viewpoint is None:
            # print(aid)
            continue
        if species not in species_dict:
            species_dict[species] = []
        if species == 'zebra_plains':
            if 'left' in viewpoint:
                species_dict[species].append(aid)
                total += 1
        if species == 'zebra_grevys':
            if 'right' in viewpoint:
                species_dict[species].append(aid)
                total += 1
        if species == 'zebra_hybrid':
            if 'right' in viewpoint:
                species_dict[species].append(aid)
                total += 1

    aid_list = species_dict[desired_species]

    metadata_dict_list = ibs.get_annot_metadata(aid_list)
    excluded_list = [
        metadata_dict.get('excluded', False) for metadata_dict in metadata_dict_list
    ]
    new_aid_list = ut.compress(aid_list, ut.not_list(excluded_list))

    return new_aid_list


@register_route('/turk/identification/graph/refer/', methods=['GET'])
def turk_identification_graph_refer(
    imgsetid, species=None, tier=1, year=2019, option=None, **kwargs
):
    ibs = current_app.ibs

    if ibs.dbname == 'ZEBRA_Kaia':
        if tier not in [0]:
            from wbia.algo.graph import mixin_loops

            mixin_loops.PRINCETON_KAIA_EDGE_LIST = ibs._princeton_kaia_filtering(
                desired_species='zebra', tier=4
            )

        assert imgsetid in [3925, 3926]

        assert species in ['zebra_plains', 'zebra_grevys']
        aid_list = ibs._princeton_kaia_filtering(
            desired_species=species, tier=tier, year=year
        )

        imageset_text = ibs.get_imageset_text(imgsetid).lower()
        annot_uuid_list = ibs.get_annot_uuids(aid_list)
        return turk_identification_graph(
            annot_uuid_list=annot_uuid_list,
            hogwild_species=species,
            creation_imageset_rowid_list=[imgsetid],
            kaia=True,
        )
    elif ibs.dbname == 'NG_Feasability' or option in ['nassau']:
        species = 'grouper_nassau'

        gid_list = ibs.get_imageset_gids(imgsetid)
        image_reviewed_list = ibs.get_image_reviewed(gid_list)
        gid_list = ut.compress(gid_list, image_reviewed_list)

        aid_list = ut.flatten(ibs.get_image_aids(gid_list))
        species_list = ibs.get_annot_species(aid_list)
        viewpoint_list = ibs.get_annot_viewpoints(aid_list)
        quality_list = ibs.get_annot_quality_texts(aid_list)

        bad_flag_list = [species_ == const.UNKNOWN for species_ in species_list]
        bad_aid_list = ut.compress(aid_list, bad_flag_list)
        if len(bad_aid_list) > 0:
            ibs.set_annot_species(bad_aid_list, [species] * len(bad_aid_list))
            species_list = ibs.get_annot_species(aid_list)

        flag_list = []
        for aid, species_, viewpoint, quality in zip(
            aid_list, species_list, viewpoint_list, quality_list
        ):
            flag = True
            if species_ != species:
                flag = False
            if viewpoint is None or 'left' not in viewpoint:
                flag = False
            if quality in ['junk', 'poor']:
                flag = False
            flag_list.append(flag)

        aid_list_ = ut.compress(aid_list, flag_list)

        imageset_text = ibs.get_imageset_text(imgsetid).lower()
        annot_uuid_list = ibs.get_annot_uuids(aid_list_)
        return turk_identification_graph(
            annot_uuid_list=annot_uuid_list,
            hogwild_species=species,
            creation_imageset_rowid_list=[imgsetid],
        )
    elif ibs.dbname == 'SSKUNK_Master' or option in ['skunk']:
        species = 'skunk_spotted'

        gid_list = ibs.get_imageset_gids(imgsetid)
        image_reviewed_list = ibs.get_image_reviewed(gid_list)
        gid_list = ut.compress(gid_list, image_reviewed_list)

        aid_list = ut.flatten(ibs.get_image_aids(gid_list))
        species_list = ibs.get_annot_species(aid_list)
        viewpoint_list = ibs.get_annot_viewpoints(aid_list)
        quality_list = ibs.get_annot_quality_texts(aid_list)

        bad_flag_list = [species_ == const.UNKNOWN for species_ in species_list]
        bad_aid_list = ut.compress(aid_list, bad_flag_list)
        if len(bad_aid_list) > 0:
            ibs.set_annot_species(bad_aid_list, [species] * len(bad_aid_list))
            species_list = ibs.get_annot_species(aid_list)

        flag_list = []
        for aid, species_, viewpoint, quality in zip(
            aid_list, species_list, viewpoint_list, quality_list
        ):
            flag = True
            if species_ != species:
                flag = False
            if viewpoint is None or 'up' not in viewpoint:
                flag = False
            if quality in ['junk', 'poor']:
                flag = False
            flag_list.append(flag)

        aid_list_ = ut.compress(aid_list, flag_list)

        imageset_text = ibs.get_imageset_text(imgsetid).lower()
        annot_uuid_list = ibs.get_annot_uuids(aid_list_)
        return turk_identification_graph(
            annot_uuid_list=annot_uuid_list,
            hogwild_species=species,
            creation_imageset_rowid_list=[imgsetid],
        )
    elif option in ['rosemary']:
        imgsetid_ = ibs.get_imageset_imgsetids_from_text('RosemaryLoopsData')

        assert imgsetid == imgsetid_
        assert species in ['zebra_plains', 'zebra_grevys']
        assert tier == 1

        gid_list = ibs.get_imageset_gids(imgsetid)
        aid_list = ut.flatten(ibs.get_image_aids(gid_list))
        aid_list_ = _zebra_annot_filtering(
            ibs, current_aids=aid_list, desired_species=species
        )

        imageset_text = ibs.get_imageset_text(imgsetid).lower()
        annot_uuid_list = ibs.get_annot_uuids(aid_list_)
        return turk_identification_graph(
            annot_uuid_list=annot_uuid_list,
            hogwild_species=species,
            creation_imageset_rowid_list=[imgsetid],
        )
    elif ibs.dbname == 'WD_Master' or option in ['wilddog']:
        imgsetid = 1

        gid_list = ibs.get_imageset_gids(imgsetid)
        image_reviewed_list = ibs.get_image_reviewed(gid_list)
        gid_list = ut.compress(gid_list, image_reviewed_list)

        aid_list = ut.flatten(ibs.get_image_aids(gid_list))
        species_list = ibs.get_annot_species(aid_list)
        viewpoint_list = ibs.get_annot_viewpoints(aid_list)
        quality_list = ibs.get_annot_quality_texts(aid_list)

        flag_list = []
        for aid, species_, viewpoint, quality in zip(
            aid_list, species_list, viewpoint_list, quality_list
        ):
            flag = True
            if species_ is None or 'wild_dog' not in species_:
                flag = False
            if viewpoint is None or 'right' not in viewpoint:
                flag = False
            if quality in ['junk', 'poor']:
                flag = False
            flag_list.append(flag)

        aid_list_ = ut.compress(aid_list, flag_list)

        imageset_text = ibs.get_imageset_text(imgsetid).lower()
        annot_uuid_list = ibs.get_annot_uuids(aid_list_)
        return turk_identification_graph(
            annot_uuid_list=annot_uuid_list,
            hogwild_species=species,
            creation_imageset_rowid_list=[imgsetid],
        )
    else:
        aid_list = ibs.get_imageset_aids(imgsetid)
        annot_uuid_list = ibs.get_annot_uuids(aid_list)

        imageset_text = ibs.get_imageset_text(imgsetid).lower()
        species = 'zebra_grevys' if 'zebra' in imageset_text else 'giraffe_reticulated'

        config = {
            'species': species,
            'threshold': 0.75,
            'enable_canonical': True,
        }
        return turk_identification_graph(
            annot_uuid_list=annot_uuid_list,
            hogwild_species=species,
            creation_imageset_rowid_list=[imgsetid],
            **config,
        )


@register_route('/turk/query/graph/v2/refer/', methods=['GET'])
def delete_query_chips_graph_v2_refer(graph_uuid):
    ibs = current_app.ibs
    ibs.delete_query_chips_graph_v2(graph_uuid=graph_uuid)
    return redirect(url_for('view_graphs'))


@register_route('/turk/identification/hardcase/', methods=['GET'])
def turk_identification_hardcase(*args, **kwargs):
    """
    CommandLine:
        python -m wbia --db PZ_MTEST --web --browser --url=/turk/identification/hardcase/
        python -m wbia --db PZ_MTEST --web --browser --url=/turk/identification/graph/

        >>> # SCRIPT
        >>> from wbia.other.ibsfuncs import *  # NOQA
        >>> import wbia
        >>> with wbia.opendb_bg_web('PZ_Master1', managed=True) as web_ibs:
        ...     resp = web_ibs.get('/turk/identification/hardcase/')

    Ignore:
        import wbia
        ibs, aids = wbia.testdata_aids('PZ_Master1', a=':species=zebra_plains')
        print(len(aids))
        infr = wbia.AnnotInference(ibs, aids=aids, autoinit='staging')
        infr.load_published()
        print(ut.repr4(infr.status()))
        infr.qt_review_loop()

        verifiers = infr.learn_evaluation_verifiers()

        verif = verifiers['match_state']
        edges = list(infr.edges())
        real = list(infr.edge_decision_from(edges))
        hardness = 1 - verif.easiness(edges, real)

    """
    import wbia

    ibs = current_app.ibs

    # HACKS
    kwargs['hardcase'] = True
    if 'annot_uuid_list' not in kwargs:
        aids = wbia.testdata_aids(ibs=ibs, a=':species=primary')
        kwargs['annot_uuid_list'] = ibs.annots(aids).uuids

    return turk_identification_graph(*args, **kwargs)


@register_route('/turk/identification/graph/', methods=['GET', 'POST'])
def turk_identification_graph(
    graph_uuid=None,
    aid1=None,
    aid2=None,
    annot_uuid_list=None,
    hardcase=None,
    view_orientation='vertical',
    view_version=1,
    hogwild=False,
    hogwild_species=None,
    creation_imageset_rowid_list=None,
    kaia=False,
    **kwargs,
):
    """
    CommandLine:
        python -m wbia.web.routes turk_identification_graph --db PZ_Master1
        python -m wbia.web.routes turk_identification_graph --db PZ_MTEST
        python -m wbia.web.routes turk_identification_graph --db testdb1 --show

        python -m wbia --db PZ_MTEST --web --browser --url=/turk/identification/graph/ --noengine

    Example:
        >>> # SCRIPT
        >>> from wbia.other.ibsfuncs import *  # NOQA
        >>> import wbia
        >>> with wbia.opendb_bg_web('testdb1', managed=True) as web_ibs:
        ...     resp = web_ibs.get('/turk/identification/graph/')
        >>> ut.quit_if_noshow()
        >>> import wbia.plottool as pt
        >>> ut.render_html(resp.content)
        >>> ut.show_if_requested()
    """
    ibs = current_app.ibs

    if ibs.dbname == 'ZEBRA_Kaia':
        kaia = True
        view_orientation = 'horizontal'

    if hogwild_species == 'None':
        hogwild_species = None

    progress = None
    hogwild_graph_uuid = None
    refer_query_uuid = None
    refer_graph_uuid_str = None
    try:
        if hogwild and aid1 is None and aid2 is None:
            try:
                hogwild_graph_uuid = graph_uuid

                if len(current_app.GRAPH_CLIENT_DICT) == 0:
                    raise ValueError(
                        'Cannot go hogwild when there are no graphs already started'
                    )

                fallback_graph_uuid_list = []
                candidate_graph_uuid_list = []
                progress_graph_uuid_list = []

                graph_uuid_list = list(current_app.GRAPH_CLIENT_DICT.keys())
                for graph_uuid_ in graph_uuid_list:
                    print('Processing %r' % (graph_uuid_,))

                    graph_client = current_app.GRAPH_CLIENT_DICT.get(graph_uuid_, None)
                    if graph_client is None:
                        print('\t ERROR: UNKNOWN graph_client')
                        continue

                    if hogwild_species is not None:
                        imagesets = graph_client.imagesets
                        if imagesets is not None:
                            imageset_id = imagesets[0]
                            imageset_text = ibs.get_imageset_text(imageset_id).lower()
                            species = (
                                'zebra_grevys'
                                if 'zebra' in imageset_text
                                else 'giraffe_reticulated'
                            )
                            if species != hogwild_species:
                                continue

                    progress_graph_uuid_list.append(graph_uuid_)

                    if graph_client.review_dict is None:
                        print('\t ERROR: FINISHED')
                        continue

                    fallback_graph_uuid_list.append(graph_uuid_)
                    print(
                        '\t len(graph_client.futures)     = %d'
                        % (len(graph_client.futures),)
                    )
                    print(
                        '\t len(graph_client.review_dict) = %d'
                        % (len(graph_client.review_dict),)
                    )
                    if (
                        len(graph_client.futures) < 15
                        and len(graph_client.review_dict) > 0
                    ):
                        candidate_graph_uuid_list.append(graph_uuid_)

                if len(fallback_graph_uuid_list) == 0:
                    raise ValueError('Cannot go hogwild when all graphs are finished')

                if len(candidate_graph_uuid_list) == 0:
                    print(
                        'WARNING: candidate_graph_uuid_list was empty, picking random from fallback_graph_uuid_list'
                    )
                    candidate_graph_uuid_list = [random.choice(fallback_graph_uuid_list)]

                print(
                    'Using Hogwild fallback_graph_uuid_list  = %r'
                    % (fallback_graph_uuid_list,)
                )
                print(
                    'Using Hogwild candidate_graph_uuid_list = %r'
                    % (candidate_graph_uuid_list,)
                )

                assert len(candidate_graph_uuid_list) > 0
                graph_uuid = random.choice(candidate_graph_uuid_list)

                progress = 0
                for graph_uuid_ in progress_graph_uuid_list:
                    graph_client, _ = ibs.get_graph_client_query_chips_graph_v2(
                        graph_uuid_
                    )
                    infr_status = graph_client.infr_status
                    if infr_status is not None:
                        cc_status = infr_status.get('cc_status', {})
                        progress += cc_status.get('num_names_max', 0)
                progress = int(progress)

                print('Using Hogwild graph_uuid = %r' % (graph_uuid,))
            except AssertionError:
                hogwild = False
                pass

        if graph_uuid is None:
            if annot_uuid_list is None:
                import wbia

                aids = wbia.testdata_aids(ibs=ibs, a=':species=primary')
                annot_uuid_list = ibs.annots(aids).uuids

            print(
                '[routes] Starting graph turk of {} annotations'.format(
                    len(annot_uuid_list)
                )
            )
            if hardcase:
                print('[routes] Graph is in hardcase-mode')
                query_config_dict = {
                    'ranking.enabled': False,
                    'autoreview.enabled': False,
                    'redun.enabled': False,
                    'queue.conf.thresh': 'absolutely_sure',
                    'algo.hardcase': True,
                }
            elif kaia:
                print('[routes] Graph is in Kaia-mode')
                query_config_dict = {
                    'autoreview.enabled': True,
                    'autoreview.prioritize_nonpos': True,
                    'inference.enabled': False,
                    'ranking.enabled': True,
                    'redun.enabled': True,
                    'redun.enforce_neg': True,
                    'redun.enforce_pos': False,
                    'redun.neg': 2,
                    'redun.pos': 2,
                }
            else:
                print('[routes] Graph is not in hardcase-mode')
                query_config_dict = {}

            query_config_dict.update(kwargs.get('query_config_dict', {}))

            print('[routes] query_config_dict = {}'.format(ut.repr3(query_config_dict)))

            graph_uuid = ibs.query_chips_graph_v2(
                annot_uuid_list=annot_uuid_list,
                query_config_dict=query_config_dict,
                creation_imageset_rowid_list=creation_imageset_rowid_list,
                **kwargs,
            )

            print(
                'Calculated graph_uuid {} from all {} annotations'.format(
                    graph_uuid, len(annot_uuid_list)
                )
            )
            # HACK probably should be a config flag instead
            if query_config_dict:
                base = url_for('turk_identification_hardcase')
            else:
                base = url_for('turk_identification_graph')
            sep = '&' if '?' in base else '?'
            url = '%s%sgraph_uuid=%s' % (base, sep, ut.to_json(graph_uuid),)
            if hardcase:
                url += '&hardcase=True'
            url = url.replace(': ', ':')
            return redirect(url)

        values = ibs.review_graph_match_config_v2(
            graph_uuid,
            aid1=aid1,
            aid2=aid2,
            view_orientation=view_orientation,
            view_version=view_version,
        )
        finished = False
    except controller_inject.WebReviewNotReadyException:
        finished = 'engine'
    except controller_inject.WebUnavailableUUIDException as ex:
        finished = 'unavailable'
        refer_query_uuid = ex.query_uuid
        refer_graph_uuid_str = 'graph_uuid=%s' % (ut.to_json(refer_query_uuid),)
        refer_graph_uuid_str = refer_graph_uuid_str.replace(': ', ':')
    except controller_inject.WebReviewFinishedException:
        finished = True
    except controller_inject.WebUnknownUUIDException:
        return redirect(url_for('turk'))

    if not finished:
        (
            edge,
            priority,
            data_dict,
            aid1,
            aid2,
            annot_uuid_1,
            annot_uuid_2,
            image_clean_src,
            image_matches_src,
            image_heatmask_src,
            server_time_start,
        ) = values

        timedelta_str = 'Unknown'
        if aid1 is not None and aid2 is not None:
            unixtime_list = ibs.get_image_unixtime(ibs.get_annot_gids([aid1, aid2]))
            if -1.0 not in unixtime_list and 0.0 not in unixtime_list:
                timedelta = abs(unixtime_list[1] - unixtime_list[0])
                secs = 60.0
                mins = 60.0 * 60.0
                hrs = 60.0 * 60.0 * 24.0
                days = 60.0 * 60.0 * 24.0 * 365.0

                if timedelta < secs:
                    timedelta_str = '%0.2f seconds' % (timedelta,)
                elif timedelta < mins:
                    timedelta /= secs
                    timedelta_str = '%0.2f minutes' % (timedelta,)
                elif timedelta < hrs:
                    timedelta /= mins
                    timedelta_str = '%0.2f hours' % (timedelta,)
                elif timedelta < days:
                    timedelta /= hrs
                    timedelta_str = '%0.2f days' % (timedelta,)
                else:
                    timedelta /= days
                    timedelta_str = '%0.2f years' % (timedelta,)

        if progress is None:
            graph_client, _ = ibs.get_graph_client_query_chips_graph_v2(graph_uuid)
            infr_status = graph_client.infr_status
            if infr_status is not None:
                cc_status = infr_status.get('cc_status', {})
                progress = cc_status.get('num_names_max', 0)

        alert = priority is not None and priority > 10.0
        data_dict.get('prob_match')
        queue_len = data_dict.get('queue_len', 0)

        match_data = ut.odict([])
        if 'evidence_decision' in data_dict:
            match_data['State'] = const.EVIDENCE_DECISION.CODE_TO_NICE[
                data_dict['evidence_decision']
            ]

        if 'match_state' in data_dict:
            probs = data_dict['match_state']
            probs_nice = ut.map_keys(const.EVIDENCE_DECISION.CODE_TO_NICE, probs)
            ut.order_dict_by(probs_nice, const.EVIDENCE_DECISION.CODE_TO_NICE.values())
            match_data['Probs'] = probs_nice
        elif 'prob_match' in data_dict:
            match_data['Probs'] = data_dict['prob_match']
        elif 'normscore' in data_dict:
            match_data['Score'] = data_dict['normscore']

        match_data['Priority'] = priority

        if 'nid_edge' in data_dict:
            match_data['Names'] = data_dict['nid_edge']

        if 'n_ccs' in data_dict:
            match_data['Connected Components'] = data_dict['n_ccs']

        # interest_config = {
        #     'aoi_two_weight_filepath': 'candidacy',
        # }
        # interest_prediction_list = ibs.depc_annot.get_property('aoi_two', [aid1, aid2], None, config=interest_config)
        # match_data['AoI Top'] = interest_prediction_list[0]
        # match_data['AoI Bottom'] = interest_prediction_list[1]

        match_data['Queue Size'] = queue_len

        if 'Probs' in match_data:
            probs_dict = match_data.pop('Probs')
            match_data['VAMP'] = {}
            if 'Positive' in probs_dict:
                match_data['VAMP']['Positive'] = probs_dict['Positive']
            if 'Negative' in probs_dict:
                match_data['VAMP']['Negative'] = probs_dict['Negative']

        review_rowid_list = ibs._get_all_review_rowids()
        identity_list = ibs.get_review_identity(review_rowid_list)
        num_auto = identity_list.count('algo:auto_clf')
        num_manual = len(review_rowid_list) - num_auto
        match_data['Reviews'] = {
            'Auto': num_auto,
            'Manual': num_manual,
        }

        match_data = ut.repr2(match_data, si=True, precision=3, kvsep='=', nobr=1)
        previous = request.args.get('previous', None)
        if previous is not None and ';' in previous:
            previous = tuple(map(int, previous.split(';')))
            assert len(previous) == 3
    else:
        progress = False
        aid1 = None
        aid2 = None
        annot_uuid_1 = None
        annot_uuid_2 = None
        image_clean_src = None
        image_matches_src = None
        previous = None
        view_orientation = None
        match_data = None
        alert = False
        timedelta_str = None

    graph_uuid_str = 'graph_uuid=%s' % (ut.to_json(graph_uuid),)
    graph_uuid_str = graph_uuid_str.replace(': ', ':')
    if kaia:
        base = url_for('submit_identification_v2_kaia')
    else:
        base = url_for('submit_identification_v2')
    callback_url = '%s?%s' % (base, graph_uuid_str,)

    hogwild_graph_uuid_str = 'graph_uuid=%s' % (ut.to_json(hogwild_graph_uuid),)
    hogwild_graph_uuid_str = hogwild_graph_uuid_str.replace(': ', ':')

    confidence_dict = const.CONFIDENCE.NICE_TO_CODE
    confidence_nice_list = confidence_dict.keys()
    confidence_text_list = confidence_dict.values()
    confidence_selected_list = [
        confidence_text == 'unspecified' for confidence_text in confidence_text_list
    ]
    confidence_list = list(
        zip(confidence_nice_list, confidence_text_list, confidence_selected_list)
    )

    if creation_imageset_rowid_list is None:
        creation_imageset_rowid_list = []

    gid1 = ibs.get_annot_gids(aid1)
    gid2 = ibs.get_annot_gids(aid2)

    imgesetid_1_list = list(
        set(ibs.get_image_imgsetids(gid1)) - set(creation_imageset_rowid_list)
    )
    imagesettext_1_list = ibs.get_imageset_text(imgesetid_1_list)
    imagesettext_1_list = [
        imagesettext_1
        for imagesettext_1 in imagesettext_1_list
        if not imagesettext_1.startswith('*')
    ]
    imagesettext_1_list_str = ', '.join(imagesettext_1_list)

    imgesetid_2_list = list(
        set(ibs.get_image_imgsetids(gid2)) - set(creation_imageset_rowid_list)
    )
    imagesettext_2_list = ibs.get_imageset_text(imgesetid_2_list)
    imagesettext_2_list = [
        imagesettext_2
        for imagesettext_2 in imagesettext_2_list
        if not imagesettext_2.startswith('*')
    ]
    imagesettext_2_list_str = ', '.join(imagesettext_2_list)

    try:
        original_filename_1_str = os.path.split(ibs.get_image_uris_original(gid1))[1]
    except Exception:
        original_filename_1_str = 'UNKNOWN'

    try:
        original_filename_2_str = os.path.split(ibs.get_image_uris_original(gid2))[1]
    except Exception:
        original_filename_2_str = 'UNKNOWN'

    # sex1, sex2 = ibs.get_annot_sex([aid1, aid2])
    age1_min, age2_min = ibs.get_annot_age_months_est_min([aid1, aid2])
    age1_max, age2_max = ibs.get_annot_age_months_est_max([aid1, aid2])
    condition1, condition2 = ibs.get_annot_qualities([aid1, aid2])

    # if sex1 == 1:
    #     sex1 = 'male'
    # elif sex1 == 0:
    #     sex1 = 'female'
    # else:
    #     sex1 = 'unknown'

    # if sex2 == 1:
    #     sex2 = 'male'
    # elif sex2 == 0:
    #     sex2 = 'female'
    # else:
    #     sex2 = 'unknown'

    if age1_min is None and age1_max == 2:
        age1 = 'age1'
    elif age1_min == 3 and age1_max == 5:
        age1 = 'age2'
    elif age1_min == 6 and age1_max == 11:
        age1 = 'age3'
    elif age1_min == 12 and age1_max == 23:
        age1 = 'age4'
    elif age1_min == 24 and age1_max == 35:
        age1 = 'age5'
    elif age1_min == 36 and age1_max is None:
        age1 = 'age6'
    elif age1_min is None and age1_max is None:
        age1 = 'unknown'
    else:
        age1 = 'unknown'

    if age2_min is None and age2_max == 2:
        age2 = 'age1'
    elif age2_min == 3 and age2_max == 5:
        age2 = 'age2'
    elif age2_min == 6 and age2_max == 11:
        age2 = 'age3'
    elif age2_min == 12 and age2_max == 23:
        age2 = 'age4'
    elif age2_min == 24 and age2_max == 35:
        age2 = 'age5'
    elif age2_min == 36 and age2_max is None:
        age2 = 'age6'
    elif age2_min is None and age2_max is None:
        age2 = 'unknown'
    else:
        age2 = 'unknown'

    if condition1 is None:
        condition1 = 0
    if condition2 is None:
        condition2 = 0

    assert age1 in ['age1', 'age2', 'age3', 'age4', 'age5', 'age6', 'unknown']
    assert age2 in ['age1', 'age2', 'age3', 'age4', 'age5', 'age6', 'unknown']
    # assert sex1 in ['male', 'female', 'unknown']
    # assert sex2 in ['male', 'female', 'unknown']
    assert 0 <= condition1 and condition1 <= 5
    assert 0 <= condition2 and condition2 <= 5

    metadata1, metadata2 = ibs.get_annot_metadata([aid1, aid2])
    comment1 = metadata1.get('turk', {}).get('match', {}).get('comment', '')
    comment2 = metadata2.get('turk', {}).get('match', {}).get('comment', '')

    try:
        edge = (
            aid1,
            aid2,
        )
        review_rowid_list = ibs.get_review_rowids_from_edges([edge])[0]
        assert len(review_rowid_list) > 0
        review_rowid = review_rowid_list[-1]
        metadata_match = ibs.get_review_metadata(review_rowid)
        comment_match = metadata_match.get('turk', {}).get('match', {}).get('comment', '')
    except Exception:
        comment_match = ''

    graph_uuid_ = '' if graph_uuid is None else str(graph_uuid)
    template_name = 'identification_kaia' if kaia else 'identification'
    return appf.template(
        'turk',
        template_name,
        match_data=match_data,
        image_clean_src=image_clean_src,
        image_matches_src=image_matches_src,
        aid1=aid1,
        aid2=aid2,
        alert=alert,
        session=False,
        progress=progress,
        timedelta_str=timedelta_str,
        refer_query_uuid=refer_query_uuid,
        refer_graph_uuid_str=refer_graph_uuid_str,
        finished=finished,
        graph_uuid=graph_uuid_,
        hogwild=hogwild,
        age1=age1,
        age2=age2,
        # sex1=sex1,
        # sex2=sex2,
        condition1=condition1,
        condition2=condition2,
        comment1=comment1,
        comment2=comment2,
        comment_match=comment_match,
        imagesettext_1_list_str=imagesettext_1_list_str,
        imagesettext_2_list_str=imagesettext_2_list_str,
        original_filename_1_str=original_filename_1_str,
        original_filename_2_str=original_filename_2_str,
        hogwild_species=hogwild_species,
        annot_uuid_1=str(annot_uuid_1),
        annot_uuid_2=str(annot_uuid_2),
        confidence_list=confidence_list,
        previous=previous,
        graph_uuid_str=graph_uuid_str,
        hogwild_graph_uuid_str=hogwild_graph_uuid_str,
        previous_version=2,
        replace_review_rowid=-1,
        view_orientation=view_orientation,
        callback_url=callback_url,
        callback_method='POST',
        EMBEDDED_CSS=None,
        EMBEDDED_JAVASCRIPT=None,
    )


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
        python -m wbia.web.app --exec-turk_quality --db PZ_Master1
        python -m wbia.web.app --exec-turk_quality --db GZ_Master1
        python -m wbia.web.app --exec-turk_quality --db GIRM_Master1

    Example:
        >>> # SCRIPT
        >>> from wbia.other.ibsfuncs import *  # NOQA
        >>> import wbia
        >>> ibs = wbia.opendb(defaultdb='testdb1')
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
        gid = ibs.get_annot_gids(aid)
        image_src = routes_ajax.annotation_src(aid)
    else:
        gid = None
        image_src = None
    imagesettext = ibs.get_imageset_text(imgsetid)
    return appf.template(
        'turk',
        'quality',
        imgsetid=imgsetid,
        src_ag=src_ag,
        dst_ag=dst_ag,
        gid=gid,
        aid=aid,
        value=value,
        image_src=image_src,
        previous=previous,
        imagesettext=imagesettext,
        progress=progress,
        finished=finished,
        display_instructions=display_instructions,
        review=review,
    )


GLOBAL_KAIA_CACHE = {}


@register_route('/turk/demographics/', methods=['GET'])
def turk_demographics(species='zebra_grevys', aid=None, **kwargs):
    with ut.Timer('turk_demographics'):
        ibs = current_app.ibs
        imgsetid = request.args.get('imgsetid', '')
        imgsetid = None if imgsetid == 'None' or imgsetid == '' else int(imgsetid)

        with ut.Timer('turk_demographics 0'):
            if ibs.dbname == 'GGR2-IBEIS':
                gid_list = ibs.get_valid_gids(imgsetid=imgsetid)
                aid_list = ut.flatten(ibs.get_image_aids(gid_list))
                aid_list = ibs.check_ggr_valid_aids(
                    aid_list, species=species, threshold=0.75
                )
            elif ibs.dbname == 'ZEBRA_Kaia':
                global GLOBAL_KAIA_CACHE
                if species not in GLOBAL_KAIA_CACHE:
                    aid_list_ = ibs.get_valid_aids()
                    nid_list_ = ibs.get_annot_nids(aid_list_)
                    flag_list = [nid > 0 for nid in nid_list_]
                    aid_list = ut.compress(aid_list_, flag_list)
                    nid_list = ut.compress(nid_list_, flag_list)
                    nid_list_ = list(set(nid_list_))
                    nid_list = list(set(nid_list))
                    print(
                        'Checking %d annotations for %d names'
                        % (len(aid_list_), len(nid_list_),)
                    )
                    print(
                        'Found %d annotations for %d names'
                        % (len(aid_list), len(nid_list),)
                    )
                    GLOBAL_KAIA_CACHE[species] = aid_list
                aid_list = GLOBAL_KAIA_CACHE[species]
            else:
                aid_list = ibs.get_valid_aids()

            reviewed_list = appf.imageset_annot_demographics_processed(ibs, aid_list)
            aid_list_ = ut.filterfalse_items(aid_list, reviewed_list)

        print('!' * 100)
        print('Demographics Progress')
        print('\t reviewed_list.count(True) = %d' % (reviewed_list.count(True),))
        print('\t len(reviewed_list)        = %d' % (len(aid_list),))
        print('!' * 100)

        try:
            progress = '%0.2f' % (100.0 * reviewed_list.count(True) / len(aid_list),)
        except ZeroDivisionError:
            progress = '0.00'

        imagesettext = None if imgsetid is None else ibs.get_imageset_text(imgsetid)
        if aid is not None:
            aid = int(aid)
        elif len(aid_list_) == 0:
            aid = None
        else:
            aid = random.choice(aid_list_)

        previous = request.args.get('previous', None)
        value_sex = ibs.get_annot_sex([aid])[0]
        if value_sex is not None:
            if value_sex >= 0:
                value_sex += 2
        else:
            value_sex = None
        value_age_min, value_age_max = list(ibs.get_annot_age_months_est([aid]))[0]
        value_age = None
        if (value_age_min == -1 or value_age_min is None) and (
            value_age_max == -1 or value_age_max is None
        ):
            value_age = 1
        if (value_age_min == 0 or value_age_min is None) and value_age_max == 2:
            value_age = 2
        elif value_age_min == 3 and value_age_max == 5:
            value_age = 3
        elif value_age_min == 6 and value_age_max == 11:
            value_age = 4
        elif value_age_min == 12 and value_age_max == 23:
            value_age = 5
        elif value_age_min == 24 and value_age_max == 35:
            value_age = 6
        elif value_age_min == 36 and (value_age_max is None or value_age_max > 36):
            value_age = 7

        review = 'review' in request.args.keys()
        finished = aid is None
        display_instructions = (
            request.cookies.get('ia-demographics_instructions_seen', 1) == 0
        )

        if not finished:
            gid = ibs.get_annot_gids(aid)
            image_src = routes_ajax.annotation_src(aid)
        else:
            gid = None
            image_src = None

        name_aid_list = None
        nid = ibs.get_annot_name_rowids(aid)
        if nid is not None:
            name_aid_list = ibs.get_name_aids(nid)
            quality_list = ibs.get_annot_qualities(name_aid_list)
            quality_text_list = ibs.get_annot_quality_texts(name_aid_list)
            viewpoint_list = ibs.get_annot_viewpoints(name_aid_list)
            name_aid_combined_list = list(
                zip(name_aid_list, quality_list, quality_text_list, viewpoint_list,)
            )
            # name_aid_combined_list.sort(key=lambda t: t[1], reverse=True)
        else:
            name_aid_combined_list = []

        region_str = 'UNKNOWN'
        if aid is not None and gid is not None:
            imgsetid_list = sorted(ibs.get_image_imgsetids(gid))
            imgset_text_list = ibs.get_imageset_text(imgsetid_list)
            imgset_text_list = [
                imgset_text
                for imgset_text in imgset_text_list
                if 'GGR Special Zone' in imgset_text
            ]
            # assert len(imgset_text_list) < 2
            # if len(imgset_text_list) == 1:
            #     region_str = imgset_text_list[0]
            region_str = ', '.join(imgset_text_list)

    return appf.template(
        'turk',
        'demographics',
        imgsetid=imgsetid,
        species=species,
        gid=gid,
        aid=aid,
        region_str=region_str,
        value_sex=value_sex,
        value_age=value_age,
        name_aid_combined_list=name_aid_combined_list,
        image_src=image_src,
        previous=previous,
        imagesettext=imagesettext,
        progress=progress,
        finished=finished,
        display_instructions=display_instructions,
        review=review,
    )


@register_route('/group_review/', methods=['GET'])
def group_review(**kwargs):
    prefill = request.args.get('prefill', '')
    if len(prefill) > 0:
        ibs = current_app.ibs
        aid_list = ibs.get_valid_aids()
        bad_species_list, bad_viewpoint_list = ibs.validate_annot_species_viewpoint_cnn(
            aid_list
        )

        GROUP_BY_PREDICTION = True
        if GROUP_BY_PREDICTION:
            grouped_dict = ut.group_items(
                bad_viewpoint_list, ut.get_list_column(bad_viewpoint_list, 3)
            )
            grouped_list = grouped_dict.values()
            regrouped_items = ut.flatten(
                ut.sortedby(grouped_list, map(len, grouped_list))
            )
            candidate_aid_list = ut.get_list_column(regrouped_items, 0)
        else:
            candidate_aid_list = [
                bad_viewpoint[0] for bad_viewpoint in bad_viewpoint_list
            ]
    elif request.args.get('aid_list', None) is not None:
        aid_list = request.args.get('aid_list', '')
        if len(aid_list) > 0:
            aid_list = aid_list.replace('[', '')
            aid_list = aid_list.replace(']', '')
            aid_list = aid_list.strip().split(',')
            candidate_aid_list = [int(aid_.strip()) for aid_ in aid_list]
        else:
            candidate_aid_list = ''
    else:
        candidate_aid_list = ''

    return appf.template(
        None,
        'group_review',
        candidate_aid_list=candidate_aid_list,
        mode_list=appf.VALID_TURK_MODES,
    )


@register_route('/sightings/', methods=['GET'])
def sightings(
    html_encode=True, complete=True, include_images=False, kaia=False, **kwargs
):
    ibs = current_app.ibs
    sightings = ibs.report_sightings_str(
        complete=complete, include_images=include_images, kaia=kaia, **kwargs
    )
    if html_encode:
        sightings = sightings.replace('\n', '<br/>')
    return sightings


@register_route('/api/', methods=['GET'], __route_prefix_check__=False)
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
            # if len(methods) > 1:
            # print('methods = %r' % (methods,))
            method = list(methods)[0]
            if method not in rule_dict.keys(**kwargs):
                rule_dict[method] = []
            rule_dict[method].append((method, url,))
    for method in rule_dict.keys(**kwargs):
        rule_dict[method].sort()
    url = '%s/api/core/dbname/' % (current_app.server_url,)
    app_auth = controller_inject.get_url_authorization(url)
    return appf.template(
        None,
        'api',
        app_url=url,
        app_name=controller_inject.GLOBAL_APP_NAME,
        app_secret=controller_inject.GLOBAL_APP_SECRET,
        app_auth=app_auth,
        rule_list=rule_dict,
    )


@register_route('/upload/', methods=['GET'], __route_authenticate__=False)
def upload(*args, **kwargs):
    # Return HTML

    embedded = dict(globals(), **locals())
    return appf.template(None, 'upload', **embedded)


@register_route('/upload/zip/', methods=['GET'])
def upload_zip(**kwargs):
    return appf.template(None, 'zip')


@register_route('/dbinfo/', methods=['GET'])
def dbinfo(**kwargs):
    try:
        ibs = current_app.ibs
        dbinfo_str = ibs.get_dbinfo_str()
    except Exception:
        dbinfo_str = ''
    dbinfo_str_formatted = '<pre>%s</pre>' % (dbinfo_str,)
    return dbinfo_str_formatted


@register_route('/counts/', methods=['GET'])
def wb_counts(**kwargs):
    fmt_str = """<p># Annotations: <b>%d</b></p>
<p># MediaAssets (images): <b>%d</b></p>
<p># MarkedIndividuals: <b>%d</b></p>
<p># Encounters: <b>%d</b></p>
<p># Occurrences: <b>%d</b></p>"""

    try:
        ibs = current_app.ibs

        aid_list = ibs.get_valid_aids()
        nid_list = ibs.get_annot_nids(aid_list)
        nid_list = [nid for nid in nid_list if nid > 0]
        gid_list = ibs.get_annot_gids(aid_list)
        imgset_id_list = ibs.get_valid_imgsetids()
        aids_list = ibs.get_imageset_aids(imgset_id_list)
        imgset_id_list = [
            imgset_id
            for imgset_id, aid_list_ in list(zip(imgset_id_list, aids_list))
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

        args = (
            num_aid,
            num_gid,
            num_nid,
            num_encounters,
            num_imgset,
        )
        counts_str = fmt_str % args
    except Exception:
        counts_str = ''
    return counts_str


@register_route('/test/counts.jsp', methods=['GET'], __route_postfix_check__=False)
def wb_counts_alias1(**kwargs):
    return wb_counts()


@register_route('/gzgc/counts.jsp', methods=['GET'], __route_postfix_check__=False)
def wb_counts_alias2(**kwargs):
    return wb_counts()


@register_route('/404/', methods=['GET'])
def error404(exception=None):
    import traceback

    exception_str = str(exception)
    traceback_str = str(traceback.format_exc())
    print('[web] %r' % (exception_str,))
    print('[web] %r' % (traceback_str,))
    return appf.template(
        None, '404', exception_str=exception_str, traceback_str=traceback_str
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
