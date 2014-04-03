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

    #
    #
    printTEST('[TEST] IMPORT_TEST_GPATHS', wait=True)
    gpath_list = __testing__.get_pyhesaff_test_gpaths(ndata=None,
                                                      zebra=True,
                                                      lena=True, jeff=True)
    gid_list = back.import_images(gpath_list=gpath_list)
    assert len(gid_list) == len(gpath_list)

    #
    #
    def add_roi(gid, bbox, theta=0.0):
        back.add_roi(gid=gid, bbox=bbox, theta=theta)

    add_roi(gid_list[0], [0, 0, 100, 100])
    add_roi(gid_list[1], [50, 50, 100, 100])
    add_roi(gid_list[2], [10, 10, 64, 64])
    #add_roi(gid_list[2], None)  # user selection
    #add_roi(None, [42, 42, 8, 8])  # back selection

    __testing__.main_loop(main_locals, rungui=RUNGUI)
    return locals()


TEST_GUI_ALL.func_name = TEST_NAME

if __name__ == '__main__':
    # For windows
    multiprocessing.freeze_support()
    test_locals = TEST_GUI_ALL()
    ibs = test_locals['ibs']
    back = test_locals['back']
    gid_list = test_locals['gid_list']
    exec(__testing__.execfunc()())
