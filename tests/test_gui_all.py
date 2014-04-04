#!/usr/bin/env python
# TODO: ADD COPYRIGHT TAG
from __future__ import print_function, division
import sys
import os
sys.path.append(os.path.expanduser('~/code/ibeis/tests'))
#-----
TEST_NAME = 'TEST_GUI_ALL'
#-----
import __testing__
import multiprocessing
import utool
from ibeis.dev import params
print, print_, printDBG, rrr, profile = utool.inject(__name__, '[%s]' % TEST_NAME)

printTEST = __testing__.printTEST

RUNGUI = utool.get_flag('--gui')


@__testing__.testcontext
def TEST_GUI_ALL():
    main_locals = __testing__.main()
    ibs = main_locals['ibs']    # IBEIS Control  # NOQA
    back = main_locals['back']  # IBEIS GUI backend

    work_dir   = params.get_workdir()
    new_dbname = 'testdb_guiall'
    new_dbdir = utool.truepath(utool.join(work_dir, new_dbname))
    ibs_dbdir = utool.truepath(ibs.dbdir)
    msg = 'must start in different dir new_dbdir=%r != ibs_dbdir=%r,' % (new_dbdir, ibs_dbdir)
    assert new_dbdir != ibs_dbdir, msg
    print('passed: ' + msg)
    utool.delete(new_dbdir)

    #
    #
    printTEST('[TEST] NEW_DATABASE')
    back.new_database(new_dbdir)
    ibs = back.ibs  # The backend has a new ibeis do not use the old one

    #
    #
    printTEST('[TEST] IMPORT_TEST_GPATHS', wait=True)
    gpath_list = __testing__.get_pyhesaff_test_gpaths(ndata=None,
                                                      zebra=True,
                                                      lena=True, jeff=True)
    gid_list = back.import_images(gpath_list=gpath_list)
    print('\n'.join('  * gid_list[%d] = %r' % (count, gid) for count, gid in enumerate(gid_list)))
    assert len(gid_list) == len(gpath_list)

    #
    #
    def add_roi(gid, bbox, theta=0.0):
        rid = back.add_roi(gid=gid, bbox=bbox, theta=theta)
        return rid

    rid1 = add_roi(gid_list[0], [0, 0, 100, 100])
    rid2 = add_roi(gid_list[1], [50, 50, 100, 100])
    rid3 = add_roi(gid_list[2], [50, 50, 64, 64])
    rid4 = add_roi(gid_list[2], [50, 50, 200, 200])
    rid5 = add_roi(gid_list[1], [0, 0, 400, 400])

    #
    #
    printTEST('[TEST] SELECT ROI / Add Chips')
    rid_list = ibs.get_valid_rids()
    print('\n'.join('  * rid_list[%d] = %r' % (count, rid) for count, rid in enumerate(rid_list)))

    print(' * rid1 = %r' % rid1)
    #print(' * rid2 = %r' % rid2)
    #print(' * rid3 = %r' % rid3)

    back.select_rid(rid_list[0])
    #back.select_rid(rid_list[2])
    #back.select_rid(rid_list[1])

    #add_roi(gid_list[2], None)  # user selection
    #add_roi(None, [42, 42, 8, 8])  # back selection
    main_locals.update(locals())
    __testing__.main_loop(main_locals, rungui=RUNGUI)
    # I'm not sure how I want to integrate that IPython stuff
    return locals()


TEST_GUI_ALL.func_name = TEST_NAME

if __name__ == '__main__':
    # For windows
    multiprocessing.freeze_support()
    test_locals = TEST_GUI_ALL()
    ibs = test_locals['ibs']
    back = test_locals['back']
    gid_list = test_locals['gid_list']
