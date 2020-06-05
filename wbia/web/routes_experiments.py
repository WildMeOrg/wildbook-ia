# -*- coding: utf-8 -*-
"""
Dependencies: flask, tornado
"""
from __future__ import absolute_import, division, print_function
from wbia.control import controller_inject
from wbia.web import appfuncs as appf
from os.path import abspath, expanduser, join
import utool as ut
import vtool as vt
import numpy as np
import cv2

(print, rrr, profile) = ut.inject2(__name__)

CLASS_INJECT_KEY, register_ibs_method = controller_inject.make_ibs_register_decorator(
    __name__
)
register_api = controller_inject.get_wbia_flask_api(__name__)
register_route = controller_inject.get_wbia_flask_route(__name__)


DBDIR_PREFIX = '/Datasets'
# DBDIR_PREFIX = '/media/hdd/work'


DB_DICT = {}
DBDIR_DICT = {
    'demo-jasonp': '/data2/wbia/DEMO2-JASONP',
    'demo-chuck': '/data2/wbia/DEMO2-CHUCK',
    'demo-hendrik': '/data2/wbia/DEMO2-HENDRIK',
    'demo-jonv': '/data2/wbia/DEMO2-JONV',
    'demo-jasonh': '/data2/wbia/DEMO2-JASONH',
    'demo-dan': '/data2/wbia/DEMO2-DAN',
    'demo-kaia': '/data2/wbia/DEMO2-KAIA',
    'demo-tanya': '/data2/wbia/DEMO2-TANYA',
    'voting': join(DBDIR_PREFIX, 'DETECT_TEAM'),
    'voting-team1': join(DBDIR_PREFIX, 'DETECT_TEAM1'),
    'voting-team2': join(DBDIR_PREFIX, 'DETECT_TEAM2'),
    'voting-team3': join(DBDIR_PREFIX, 'DETECT_TEAM3'),
    'voting-team4': join(DBDIR_PREFIX, 'DETECT_TEAM4'),
    'voting-team5': join(DBDIR_PREFIX, 'DETECT_TEAM5'),
}


GLOBAL_AOI_VALUES = None
GLOBAL_AOI_DICT = None
GLOBAL_IMAGE_UUID_LIST = None
GLOBAL_ANNOT_UUID_LIST = None


@register_route('/experiments/', methods=['GET'])
def view_experiments(**kwargs):
    return appf.template('experiments')


def experiment_init_db(tag):
    import wbia

    if tag in DBDIR_DICT:
        dbdir = abspath(expanduser(DBDIR_DICT[tag]))
        DB_DICT[tag] = wbia.opendb(dbdir=dbdir, web=False)
    return DB_DICT.get(tag, None)


@register_route('/experiments/ajax/image/src/<tag>/', methods=['GET'])
def experiments_image_src(tag=None, **kwargs):
    tag = tag.strip().split('-')
    db = tag[0]
    gid = int(tag[1])

    ibs = experiment_init_db(db)
    config = {
        'thumbsize': 800,
    }
    gpath = ibs.get_image_thumbpath(gid, ensure_paths=True, **config)

    # Load image
    image = vt.imread(gpath, orient='auto')
    image = appf.resize_via_web_parameters(image)
    return appf.embed_image_html(image, target_width=None)


@register_route('/experiments/interest/', methods=['GET'])
def experiments_interest(dbtag1='demo-jasonp', dbtag2='demo-chuck', **kwargs):
    from uuid import UUID
    from wbia.other.detectfuncs import general_overlap, general_parse_gt

    dbtag1 = str(dbtag1)
    dbtag2 = str(dbtag2)
    ibs1 = experiment_init_db(dbtag1)
    ibs2 = experiment_init_db(dbtag2)
    dbdir1 = ibs1.dbdir
    dbdir2 = ibs2.dbdir

    gid_list1 = ibs1.get_valid_gids(reviewed=1)
    gid_list2 = ibs2.get_valid_gids(reviewed=1)

    gt_dict1 = general_parse_gt(ibs1, gid_list1)
    gt_dict2 = general_parse_gt(ibs2, gid_list2)

    uuid_list1 = sorted(map(str, ibs1.get_image_uuids(gid_list1)))
    uuid_list2 = sorted(map(str, ibs2.get_image_uuids(gid_list2)))

    gid_pair_list = []
    index1, index2 = 0, 0
    stats_global = {
        'disagree_interest1': 0,
        'disagree_interest2': 0,
        'annot1': 0,
        'annot2': 0,
    }
    while index1 < len(uuid_list1) or index2 < len(uuid_list2):
        uuid1 = UUID(uuid_list1[index1]) if index1 < len(uuid_list1) else None
        uuid2 = UUID(uuid_list2[index2]) if index2 < len(uuid_list2) else None

        if uuid1 is None and uuid2 is None:
            break

        gid1 = ibs1.get_image_gids_from_uuid(uuid1)
        gid2 = ibs2.get_image_gids_from_uuid(uuid2)

        print('%s %s' % (index1, index2,))
        stats = None
        if uuid1 is not None and uuid2 is not None:
            if uuid1 == uuid2:
                gt_list1 = gt_dict1[uuid1]
                gt_list2 = gt_dict2[uuid2]

                stats_global['annot1'] += len(gt_list1)
                stats_global['annot2'] += len(gt_list2)

                overlap = general_overlap(gt_list1, gt_list2)
                if 0 in overlap.shape:
                    index_list1 = []
                    index_list2 = []
                else:
                    index_list1 = np.argmax(overlap, axis=1)
                    index_list2 = np.argmax(overlap, axis=0)

                pair_list1 = set(enumerate(index_list1))
                pair_list2 = set(enumerate(index_list2))
                pair_list2 = set([_[::-1] for _ in pair_list2])
                pair_union = pair_list1 | pair_list2
                pair_intersect = pair_list1 & pair_list2
                pair_diff_sym = pair_list1 ^ pair_list2
                pair_diff1 = pair_list1 - pair_list2
                pair_diff2 = pair_list2 - pair_list1

                message_list = []
                if len(gt_list1) > 0 and len(gt_list2) == 0:
                    message_list.append('Jason has annotations, Chuck none')
                if len(gt_list1) == 0 and len(gt_list2) > 0:
                    message_list.append('Chuck has annotations, Jason none')
                if len(pair_diff1) > 0 and len(pair_diff2) == 0:
                    message_list.append('Jason has additional annotations')
                if len(pair_diff1) == 0 and len(pair_diff2) > 0:
                    message_list.append('Chuck has additional annotations')
                if len(pair_diff1) > 0 and len(pair_diff2) > 0:
                    message_list.append('Assignment mismatch')

                disagree = 0
                for index1_, index2_ in pair_intersect:
                    gt1 = gt_list1[index1_]
                    gt2 = gt_list2[index2_]
                    interest1 = gt1['interest']
                    interest2 = gt2['interest']

                    if interest1 != interest2:
                        disagree += 1
                        if interest1 > interest2:
                            stats_global['disagree_interest1'] += 1
                        if interest2 > interest1:
                            stats_global['disagree_interest2'] += 1

                if disagree > 0:
                    message_list.append('Interest mismatch')

                stats = {
                    'num_annot1': len(gt_list1),
                    'num_annot2': len(gt_list2),
                    'num_interest1': len([_ for _ in gt_list1 if _['interest']]),
                    'num_interest2': len([_ for _ in gt_list2 if _['interest']]),
                    'conflict': len(message_list) > 0,
                    'message': '<br/>'.join(message_list),
                }
            else:
                if uuid1 < uuid2:
                    gid2 = None
                else:
                    gid1 = None

        gid_pair_list.append((gid1, gid2, stats))
        if gid1 is not None:
            index1 += 1
        if gid2 is not None:
            index2 += 1

    embed = dict(globals(), **locals())
    return appf.template('experiments', 'interest', **embed)


def voting_uuid_list(ibs, team_list):
    blacklist = []
    image_uuid_list = ibs.get_image_uuids(ibs.get_valid_gids())
    # image_uuid_list = image_uuid_list[:100]
    annot_uuid_list = ut.flatten(
        ibs.get_image_annot_uuids(ibs.get_image_gids_from_uuid(image_uuid_list))
    )
    for team in team_list:
        print('Checking team %r' % (team,))
        try:
            gid_list = team.get_image_gids_from_uuid(image_uuid_list)
            assert None not in gid_list
        except AssertionError:
            zipped = zip(image_uuid_list, gid_list)
            blacklist += [image_uuid for image_uuid, gid in zipped if gid is None]
        try:
            aid_list = team.get_annot_aids_from_uuid(annot_uuid_list)
            assert None not in aid_list
        except AssertionError:
            zipped = zip(annot_uuid_list, aid_list)
            blacklist += [
                ibs.get_image_uuids(
                    ibs.get_annot_image_rowids(ibs.get_annot_aids_from_uuid(annot_uuid))
                )
                for annot_uuid, aid in zipped
                if aid is None
            ]
    blacklist = list(set(blacklist))
    assert None not in blacklist
    print('Blacklisted %d / %d' % (len(blacklist), len(image_uuid_list),))
    image_uuid_list = list(set(image_uuid_list) - set(blacklist))
    annot_uuid_list = ut.flatten(
        ibs.get_image_annot_uuids(ibs.get_image_gids_from_uuid(image_uuid_list))
    )
    return image_uuid_list, annot_uuid_list


def voting_data(
    method=3,
    option='inclusive',
    species='all',
    team1=True,
    team2=True,
    team3=True,
    team4=True,
    team5=True,
):
    global GLOBAL_AOI_VALUES, GLOBAL_AOI_DICT, GLOBAL_ANNOT_UUID_LIST

    method = int(method)
    method_list = [1, 2, 3, 4, 5]
    assert method in method_list
    assert option in ['inclusive', 'exclusive']
    assert species in ['all', 'giraffe', 'kenya', 'seaturtle', 'whalefluke', 'zebra']

    enabled_list = [team1, team2, team3, team4, team5]
    assert False not in [enabled in [True, False] for enabled in enabled_list]

    method_ = enabled_list.count(True)
    method = min(method, method_)

    aoi_values = (method, option, species, team1, team2, team3, team4, team5)

    if GLOBAL_AOI_VALUES == aoi_values and GLOBAL_AOI_DICT is not None:
        return GLOBAL_AOI_DICT

    ibs, team_list = experiments_voting_initialize(enabled_list)

    if species == 'all':
        species_list = [
            'giraffe_masai',
            'giraffe_reticulated',
            'turtle_sea',
            'whale_fluke',
            'zebra_grevys',
            'zebra_plains',
        ]
    elif species == 'giraffe':
        species_list = [
            'giraffe_amasai',
            'giraffe_reticulated',
        ]
    elif species == 'kenya':
        species_list = [
            'giraffe_amasai',
            'giraffe_reticulated',
            'zebra_grevys',
            'zebra_plains',
        ]
    elif species == 'seaturtle':
        species_list = [
            'turtle_sea',
        ]
    elif species == 'whalefluke':
        species_list = [
            'whale_fluke',
        ]
    elif species == 'zebra':
        species_list = [
            'zebra_grevys',
            'zebra_plains',
        ]
    else:
        raise AssertionError

    species_set = set(species_list)

    aoi_dict = {}
    for annot_uuid in GLOBAL_ANNOT_UUID_LIST:
        aid = ibs.get_annot_aids_from_uuid(annot_uuid)
        species = ibs.get_annot_species_texts(aid)

        if species not in species_set:
            continue

        flag_list = [
            team.get_annot_interest(team.get_annot_aids_from_uuid(annot_uuid))
            for team in team_list
        ]
        count = flag_list.count(True)
        count -= method

        if option == 'inclusive':
            aoi = count >= 0
        elif option == 'exclusive':
            aoi = count == 0
        assert aoi in [True, False]

        aoi_dict[annot_uuid] = aoi

    GLOBAL_AOI_VALUES = aoi_values
    GLOBAL_AOI_DICT = aoi_dict

    return aoi_dict


@register_api(
    '/experiments/ajax/voting/count/', methods=['GET'], __api_plural_check__=False
)
def experiments_voting_counts(ibs, **kwargs):
    aoi_dict = voting_data(**kwargs)

    # Totals
    flag_list = list(aoi_dict.values())
    count = flag_list.count(True)
    total = len(flag_list)
    if total == 0:
        total_str = 'Undefined'
    else:
        total_str = '%0.2f' % (100.0 * float(count) / total,)

    stage1_count_list = [count, total_str, total]

    # Image Totals
    ibs, team_list = experiments_voting_initialize()

    gid_dict = {}
    for annot_uuid in aoi_dict:
        aid = ibs.get_annot_aids_from_uuid(annot_uuid)
        gid = ibs.get_annot_gids(aid)
        if gid not in gid_dict:
            gid_dict[gid] = [0, 0]
        gid_dict[gid][1] += 1
        if aoi_dict[annot_uuid]:
            gid_dict[gid][0] += 1

    count_list = []
    precentage_list = []
    for gid in gid_dict:
        count = gid_dict[gid][0]
        total = gid_dict[gid][1]
        precentage = float(count) / float(total)
        count_list.append(count)
        precentage_list.append(precentage)

    total = len(count_list)
    if total == 0:
        count_str = 'Undefined'
        deviation_str = 'Undefined'
        percentage_str = 'Undefined'
    else:
        count_list = np.array(count_list)
        avg = np.mean(count_list)
        std = np.std(count_list)
        count_str = '%0.2f' % (avg,)
        deviation_str = '%0.2f' % (std,)
        percentage_str = '%0.2f' % (100.0 * sum(precentage_list) / total,)

    stage2_count_list = [count_str, deviation_str, percentage_str]

    return stage1_count_list, stage2_count_list


@register_api(
    '/experiments/ajax/voting/variance/', methods=['GET'], __api_plural_check__=False
)
def experiments_voting_variance(ibs, team_index, **kwargs):
    aoi_dict = voting_data(**kwargs)
    team_index = int(team_index)
    assert team_index in [0, 1, 2, 3, 4]

    ibs, team_list = experiments_voting_initialize()
    team = team_list[team_index]

    incorrect = 0
    total = 0
    for annot_uuid in aoi_dict:
        aid = team.get_annot_aids_from_uuid(annot_uuid)
        interest = team.get_annot_interest(aid)
        if interest != aoi_dict[annot_uuid]:
            incorrect += 1
        total += 1

    if total == 0:
        accuracy_str = 'Undefined'
    else:
        correct = total - incorrect
        accuracy_str = '%0.02f' % (100.0 * (correct / total),)
    return team_index, incorrect, accuracy_str


def _normalize_image(image):
    image *= 255.0
    image[image < 0.0] = 0
    image[image > 255.0] = 255.0
    image = np.around(image)
    image = image.astype(np.uint8)

    image = cv2.merge((image, image, image))
    image = cv2.resize(image, (400, 400), interpolation=cv2.INTER_NEAREST)
    return image


@register_api(
    '/experiments/ajax/voting/center/src/', methods=['GET'], __api_plural_check__=False
)
def experiments_voting_center_src(ibs, aoi=False, **kwargs):
    aoi_dict = voting_data(**kwargs)
    ibs, team_list = experiments_voting_initialize()

    image = np.zeros((100, 100, 1), dtype=np.float32)
    for annot_uuid in aoi_dict:
        if aoi and not aoi_dict[annot_uuid]:
            continue
        aid = ibs.get_annot_aids_from_uuid(annot_uuid)
        gid = ibs.get_annot_gids(aid)
        width, height = ibs.get_image_sizes(gid)
        (x, y, w, h) = ibs.get_annot_bboxes(aid)
        cx = x + (w / 2.0)
        cy = y + (h / 2.0)
        cx /= width
        cy /= height
        cx = int(np.round(cx * 100.0))
        cy = int(np.round(cy * 100.0))
        if 0 <= cx and cx < 100 and 0 <= cy and cy < 100:
            image[cx, cy] += 1.0

    maximum = np.max(image)
    image /= maximum
    image = _normalize_image(image)

    # Load image
    return maximum, appf.embed_image_html(image, target_width=None)


@register_api(
    '/experiments/ajax/voting/area/src/', methods=['GET'], __api_plural_check__=False
)
def experiments_voting_area_src(ibs, aoi=False, **kwargs):
    aoi_dict = voting_data(**kwargs)
    ibs, team_list = experiments_voting_initialize()

    image = np.zeros((100, 100, 1), dtype=np.float32)
    for annot_uuid in aoi_dict:
        if aoi and not aoi_dict[annot_uuid]:
            continue
        aid = ibs.get_annot_aids_from_uuid(annot_uuid)
        gid = ibs.get_annot_gids(aid)
        width, height = ibs.get_image_sizes(gid)
        (x, y, w, h) = ibs.get_annot_bboxes(aid)
        x0 = x
        y0 = y
        x1 = x + w
        y1 = y + h
        x0 /= width
        y0 /= height
        x1 /= width
        y1 /= height
        x0 = min(max(x0, 0.0), 1.0)
        y0 = min(max(y0, 0.0), 1.0)
        x1 = min(max(x1, 0.0), 1.0)
        y1 = min(max(y1, 0.0), 1.0)
        x0 = int(np.around(x0 * 100.0))
        y0 = int(np.around(y0 * 100.0))
        x1 = int(np.around(x1 * 100.0))
        y1 = int(np.around(y1 * 100.0))
        image[x0:x1, y0:y1] += 1.0

    maximum = np.max(image)
    image /= maximum
    image = _normalize_image(image)

    # Load image
    return maximum, appf.embed_image_html(image, target_width=None)


@register_api(
    '/experiments/ajax/voting/bbox/metrics/', methods=['GET'], __api_plural_check__=False,
)
def experiments_voting_bbox_width(ibs, **kwargs):
    aoi_dict = voting_data(**kwargs)
    ibs, team_list = experiments_voting_initialize()

    bins = 10.0
    key_list = list(range(int(bins + 1)))
    histogram_dict = {
        'keys': key_list,
        'width': [{index: 0 for index in key_list} for _ in range(2)],
        'height': [{index: 0 for index in key_list} for _ in range(2)],
        'area': [{index: 0 for index in key_list} for _ in range(2)],
    }
    for annot_uuid in aoi_dict:
        aid = ibs.get_annot_aids_from_uuid(annot_uuid)
        gid = ibs.get_annot_gids(aid)
        width, height = ibs.get_image_sizes(gid)
        (x, y, w, h) = ibs.get_annot_bboxes(aid)
        w /= width
        h /= height
        a = w * h
        w = min(max(w, 0.0), 1.0)
        h = min(max(h, 0.0), 1.0)
        a = min(max(a, 0.0), 1.0)
        w = int(np.around(w * bins))
        h = int(np.around(h * bins))
        a = int(np.around(a * bins))
        histogram_dict['width'][0][w] += 1
        histogram_dict['height'][0][h] += 1
        histogram_dict['area'][0][a] += 1
        if aoi_dict[annot_uuid]:
            histogram_dict['width'][1][w] += 1
            histogram_dict['height'][1][h] += 1
            histogram_dict['area'][1][a] += 1

    for histogram_key in histogram_dict:
        if histogram_key == 'keys':
            continue
        histogram_list = histogram_dict[histogram_key]
        for histogram_index in range(len(histogram_list)):
            histogam = histogram_list[histogram_index]
            histogram_list[histogram_index] = [histogam[key] for key in key_list]

    return int(bins), histogram_dict


def experiments_voting_initialize(enabled_list=None):
    global GLOBAL_IMAGE_UUID_LIST, GLOBAL_ANNOT_UUID_LIST

    ibs = experiment_init_db('voting')
    assert ibs is not None

    ibs1 = experiment_init_db('voting-team1')
    ibs2 = experiment_init_db('voting-team2')
    ibs3 = experiment_init_db('voting-team3')
    ibs4 = experiment_init_db('voting-team4')
    ibs5 = experiment_init_db('voting-team5')
    team_list = [ibs1, ibs2, ibs3, ibs4, ibs5]
    assert None not in team_list

    if enabled_list is not None:
        assert len(team_list) == 5
        assert False not in [enabled in [True, False] for enabled in enabled_list]
        team_list = [team for team, enabled in zip(team_list, enabled_list) if enabled]

    if None in [GLOBAL_IMAGE_UUID_LIST, GLOBAL_ANNOT_UUID_LIST]:
        image_uuid_list, annot_uuid_list = voting_uuid_list(ibs, team_list)
        GLOBAL_IMAGE_UUID_LIST = image_uuid_list
        GLOBAL_ANNOT_UUID_LIST = annot_uuid_list

    return ibs, team_list


@register_route('/experiments/voting/', methods=['GET'])
def experiments_voting(**kwargs):

    experiments_voting_initialize()
    voting_data()

    embed = dict(globals(), **locals())
    return appf.template('experiments', 'voting', **embed)


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
