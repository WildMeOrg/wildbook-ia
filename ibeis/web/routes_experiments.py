# -*- coding: utf-8 -*-
"""
Dependencies: flask, tornado
"""
from __future__ import absolute_import, division, print_function
from ibeis.control import controller_inject
from ibeis.web import appfuncs as appf
from os.path import abspath, expanduser
import utool as ut
import vtool as vt
import numpy as np

CLASS_INJECT_KEY, register_ibs_method = (
    controller_inject.make_ibs_register_decorator(__name__))
register_route = controller_inject.get_ibeis_flask_route(__name__)


DB_DICT = {}
# DBDIR_ROOT = '~/Desktop/'
DBDIR_ROOT = '/data2/ibeis/'
DBDIR_DICT = {
    'jasonp':  'DEMO2-JASONP',
    'chuck':   'DEMO2-CHUCK',
    'hendrik': 'DEMO2-HENDRIK',
    'jonv':    'DEMO2-JONV',
    'jasonh':  'DEMO2-JASONH',
    'dan':     'DEMO2-DAN',
    'kaia':    'DEMO2-KAIA',
}


@register_route('/experiments/', methods=['GET'])
def view_experiments(**kwargs):
    return appf.template('experiments')


def experiment_init_db(tag):
    import ibeis
    if tag in DBDIR_DICT:
        dbdir = abspath(expanduser(DBDIR_ROOT + DBDIR_DICT[tag]))
        DB_DICT[tag] = ibeis.opendb(dbdir=dbdir, web=False)
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
def experiments_interest(dbtag1='jasonp', dbtag2='chuck', **kwargs):
    from uuid import UUID
    from ibeis.other.detectfuncs import general_overlap, general_parse_gt

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

        ut.embed()

        if uuid1 is None and uuid2 is None:
            break

        gid1 = ibs1.get_image_gids_from_uuid(uuid1)
        gid2 = ibs2.get_image_gids_from_uuid(uuid2)

        print('%s %s' % (index1, index2, ))
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

        gid_pair_list.append( (gid1, gid2, stats) )
        if gid1 is not None:
            index1 += 1
        if gid2 is not None:
            index2 += 1

    embed = dict(globals(), **locals())
    return appf.template('experiments', 'interest', **embed)


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
