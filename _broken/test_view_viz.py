#!/usr/bin/env python2.7
# TODO: ADD COPYRIGHT TAG
from __future__ import absolute_import, division, print_function
import multiprocessing
import utool
from ibeis import viz
from plottool import draw_func2 as df2
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[TEST_VIZ]')


def TEST_VIZ(ibs):
    valid_gids = ibs.get_valid_gids()
    valid_aids = ibs.get_valid_aids()
    print('len(valid_aids) = %r' % len(valid_aids))
    print('len(valid_gids) = %r' % len(valid_gids))
    assert len(valid_gids) > 0, 'database images cannot be empty for test'
    gindex = 1
    gid = valid_gids[gindex]
    aid_list = ibs.get_image_aids(gid)
    rindex = 0
    aid = aid_list[rindex]
    qaid = aid
    aids = aid_list[1:3]
    aid = aid_list[-1]

    try:
        qres = ibs._query_chips4([qaid], valid_aids)[qaid]
        print(qres)
        top_aids = qres.get_top_aids(ibs)
        assert len(top_aids) > 0, 'Results seems to be empty'
        aid2 = top_aids[0]  # 294
        query_failed = False
    except Exception as ex:
        query_failed = True
        utool.printex(ex, 'QUERY FAILED!')
        raise

    #----------------------
    #print('Show Image')
    viz.show_image(ibs, gid, aids=aids, fnum=1)
    df2.set_figtitle('Show Image')

    #----------------------
    #print('Show Chip')
    kpts_kwgs = dict(ell=True, ori=True, rect=True,
                     eig=True, pts=False, kpts_subset=10)
    viz.show_chip(ibs, aid, in_image=False, fnum=2, **kpts_kwgs)
    df2.set_figtitle('Show Chip (normal)')
    viz.show_chip(ibs, aid, in_image=True, fnum=3, **kpts_kwgs)
    df2.set_figtitle('Show Chip (in_image)')

    #----------------------
    if not query_failed:
        print('Show Query')
        viz.show_matches(ibs, qres, aid2, fnum=4)
        df2.set_figtitle('Show Chipres')

        viz.show_qres(ibs, qres, fnum=5)
        df2.set_figtitle('Show QRes')

    ##----------------------
    print('return test locals')
    return locals()


if __name__ == '__main__':
    import ibeis
    multiprocessing.freeze_support()  # For windows
    main_locals = ibeis.main(defaultdb='testdb1', gui=False)
    ibs = main_locals['ibs']    # IBEIS Control
    test_locals = utool.run_test(TEST_VIZ, ibs)
    execstr = utool.execstr_dict(test_locals, 'test_locals')
    exec(execstr)
    import sys
    if '--noshow' not in sys.argv:
        exec(df2.present())
    exec(utool.ipython_execstr())
