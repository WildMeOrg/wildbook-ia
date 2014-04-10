#!/usr/bin/env python
# TODO: ADD COPYRIGHT TAG
from __future__ import print_function, division
#-----
TEST_NAME = 'TEST_VIZ'
#-----
import sys
sys.argv.append('--nogui')
import __testing__
import multiprocessing
import utool
from ibeis.view import viz_image, viz_chip, viz_helpers, viz_matches
from drawtool import draw_func2 as df2
from ibeis.model.jon_recognition import matching_functions as mf
from ibeis.model.jon_recognition import QueryResult, QueryRequest  # NOQA
from ibeis.model.jon_recognition.matching_functions import _apply_filter_scores, progress_func, _fix_fmfsfk  # NOQA
import build_query
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[%s]' % TEST_NAME)
printTEST = __testing__.printTEST


def myreload():
    viz_image.rrr()
    viz_chip.rrr()
    viz_helpers.rrr()
    build_query.rrr()
    mf.rrr()


@__testing__.testcontext
def TEST_VIZ():
    main_locals = __testing__.main(defaultdb='test_big_ibeis')
    ibs = main_locals['ibs']    # IBEIS Control  # NOQA

    valid_gids = ibs.get_valid_gids()
    valid_rids = ibs.get_valid_rids()

    printTEST('''
    * len(valid_rids) = %r
    * len(valid_gids) = %r
    ''' % (len(valid_rids), len(valid_gids)))
    assert len(valid_gids) > 0, 'database images cannot be empty for test'
    gid = valid_gids[0]
    rid_list = ibs.get_rids_in_gids(gid)
    cid_list = ibs.get_roi_cids(rid_list)
    cid = cid_list[min(len(cid_list) - 1, 2)]

    #----------------------
    #printTEST('Show Image')
    sel_rids = rid_list[1:3]
    viz_image.show_image(ibs, gid, sel_rids=sel_rids, fnum=1)

    #----------------------
    #printTEST('Show Chip')
    viz_chip.show_chip(ibs, cid, in_image=False, fnum=2)
    viz_chip.show_chip(ibs, cid, in_image=True, fnum=3)

    #----------------------
    printTEST('Show Query')
    cid1 = cid
    qcids = [cid1]

    qcid2_qres = ibs.query_database(qcids)
    qres = qcid2_qres[cid1]
    top_cids = qres.get_top_cids(ibs)
    assert len(top_cids) > 0, 'there does not seem to be results'
    cid2 = top_cids[0]  # 294
    viz_matches.show_chipres(ibs, qres, cid2, fnum=4)

    ##----------------------
    df2.present(wh=1000)
    main_locals.update(locals())
    __testing__.main_loop(main_locals)
    printTEST('return test locals')
    return main_locals

TEST_VIZ.func_name = TEST_NAME


if __name__ == '__main__':
    multiprocessing.freeze_support()  # For windows
    test_locals = TEST_VIZ()
    if utool.get_flag('--cmd2'):
        exec(utool.execstr_dict(test_locals, 'test_locals'))
        exec(utool.execstr_embed())
