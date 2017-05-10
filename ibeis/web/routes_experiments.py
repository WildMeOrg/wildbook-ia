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


CLASS_INJECT_KEY, register_ibs_method = (
    controller_inject.make_ibs_register_decorator(__name__))
register_route = controller_inject.get_ibeis_flask_route(__name__)


DB_DICT = {}
DBDIR_DICT = {
    '1': '~/Desktop/JASON',
    '2': '~/Desktop/CHUCK',
}


@register_route('/experiments/', methods=['GET'])
def view_experiments(**kwargs):
    return appf.template('experiments')


def experiment_init_db(tag):
    import ibeis
    if tag in DBDIR_DICT:
        dbdir = abspath(expanduser(DBDIR_DICT[tag]))
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
def experiments_interest(**kwargs):
    from uuid import UUID

    ibs1 = experiment_init_db('1')
    ibs2 = experiment_init_db('2')
    dbdir1 = ibs1.dbdir
    dbdir2 = ibs2.dbdir

    gid_list1 = ibs1.get_valid_gids()
    gid_list2 = ibs2.get_valid_gids()

    uuid_list1 = sorted(map(str, ibs1.get_image_uuids(gid_list1)))
    uuid_list2 = sorted(map(str, ibs2.get_image_uuids(gid_list2)))

    gid_pair_list = []
    index1, index2 = 0, 0
    while index1 < len(uuid_list1) or index2 < len(uuid_list2):
        uuid1 = UUID(uuid_list1[index1])
        uuid2 = UUID(uuid_list2[index2])
        gid1 = ibs1.get_image_gids_from_uuid(uuid1)
        gid2 = ibs2.get_image_gids_from_uuid(uuid2)

        metadata = { 'test': 1 }
        if uuid1 == uuid2:
            pass
        elif uuid1 < uuid2:
            gid2 = None
        else:
            gid1 = None

        gid_pair_list.append( (gid1, gid2, metadata) )
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
