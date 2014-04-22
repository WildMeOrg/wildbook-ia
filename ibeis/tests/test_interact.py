#!/usr/bin/env python
# TODO: ADD COPYRIGHT TAG
from __future__ import absolute_import, division, print_function
#-----
TEST_NAME = 'TEST_INTERACT'
#-----
import sys
sys.argv.append('--nogui')
import __testing__
import multiprocessing
import utool
from ibeis import interact  # NOQA
from plottool import draw_func2 as df2
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[%s]' % TEST_NAME)
printTEST = __testing__.printTEST


@__testing__.testcontext2(TEST_NAME)
def TEST_INTERACT():
    main_locals = __testing__.main(defaultdb='test_big_ibeis')
    ibs = main_locals['ibs']    # IBEIS Control  # NOQA

    valid_gids = ibs.get_valid_gids()
    valid_rids = ibs.get_valid_rids()

    printTEST('''
    * len(valid_rids) = %r
    * len(valid_gids) = %r
    ''' % (len(valid_rids), len(valid_gids)))
    assert len(valid_gids) > 0, 'database images cannot be empty for test'
    gindex = int(utool.get_arg('--gx', default=0))
    cindex = int(utool.get_arg('--rx', default=0))
    gid = valid_gids[gindex]
    rid_list = ibs.get_image_rois(gid)
    rid = rid_list[cindex]

    #----------------------
    #printTEST('Show Image')
    sel_rids = rid_list[1:3]
    interact.ishow_image(ibs, gid, sel_rids=sel_rids, fnum=1)

    #----------------------
    #printTEST('Show Chip')
    interact.ishow_chip(ibs, rid, in_image=False, fnum=2)
    #interact.ishow_chip(ibs, rid, in_image=True, fnum=3)

    #----------------------
    #printTEST('Show Query')
    #rid1 = rid
    #qcid2_qres = ibs.query_database([rid1])
    #qres = qcid2_qres.values()[0]
    #top_cids = qres.get_top_cids(ibs)
    #assert len(top_cids) > 0, 'there does not seem to be results'
    #cid2 = top_cids[0]  # 294
    #viz.show_matches(ibs, qres, cid2, fnum=4)

    #viz.show_qres(ibs, qres, fnum=5)

    ##----------------------
    main_locals.update(locals())
    __testing__.main_loop(main_locals)
    printTEST('return test locals')
    return main_locals


if __name__ == '__main__':
    multiprocessing.freeze_support()  # For windows
    test_locals = TEST_INTERACT()
    df2.present()
    exec(test_locals['execstr'])
