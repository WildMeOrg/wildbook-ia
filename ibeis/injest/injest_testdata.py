#!/usr/bin/env python
# TODO: ADD COPYRIGHT TAG
from __future__ import absolute_import, division, print_function
from os.path import join
import ibeis
import utool
from ibeis.injest.injest_named_images import injest_named_images
from vtool.tests import grabdata


def injest_testdata():
    print('INJEST TESTDATA')
    # Clean up testdata in work directory
    workdir = ibeis.sysres.get_workdir()
    testdb1 = join(workdir, 'testdb1')
    utool.delete(testdb1)
    # Copy testdata images
    testdata_dir = grabdata.get_testdata_dir()
    utool.copy(testdata_dir, testdb1)
    # Create IBEIS database
    main_locals = ibeis.main(dbdir=testdb1, gui=False)
    ibs = main_locals['ibs']
    fmtkey = 'testdata'
    # Injest images
    injest_named_images(ibs, testdb1, fmtkey)
    # TODO: Add correct ROIS here

    # Print to show success
    ibs.print_name_table()
    ibs.print_image_table()
    ibs.print_roi_table()

if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()  # win32
    injest_testdata()
