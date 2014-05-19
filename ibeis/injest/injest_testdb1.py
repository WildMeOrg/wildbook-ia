#!/usr/bin/env python
# TODO: ADD COPYRIGHT TAG
from __future__ import absolute_import, division, print_function
import ibeis
from ibeis.injest import injest_database
from ibeis.dev import ibsfuncs


def injest_testdb1():
    ibs = injest_database.injest_standard_database('testdb1')
    #print('INJEST testdb1')
    ## Clean up testdata in work directory
    #workdir = ibeis.sysres.get_workdir()
    #testdb1 = join(workdir, 'testdb1')
    #utool.delete(testdb1)
    ## Copy testdata images
    #testdata_dir = grabdata.get_testdata_dir()
    #utool.copy(testdata_dir, testdb1)
    ## Create IBEIS database
    #main_locals = ibeis.main(dbdir=testdb1, gui=False)
    #ibs = main_locals['ibs']
    #fmtkey = '{name:*}[id:d].{ext}'
    ## Injest images
    #injest_named_images(ibs, testdb1, fmtkey)
    ## TODO: Add correct ROIS here

    ## TestData unixtimes
    #gid_list = np.array(ibs.get_valid_gids())

    #unixtimes_even = (gid_list[0::2] + 100).tolist()
    #unixtimes_odd  = (gid_list[1::2] + 9001).tolist()
    #unixtime_list = unixtimes_even + unixtimes_odd
    #ibs.set_image_unixtime(gid_list, unixtime_list)

    ## Print to show success
    ##ibs.print_name_table()
    ##ibs.print_image_table()
    ##ibs.print_roi_table()

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()  # win32
    injest_testdb1()
