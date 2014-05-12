#!/usr/bin/env python
# TODO: ADD COPYRIGHT TAG
from __future__ import absolute_import, division, print_function
import sys
import os
from os.path import expanduser, join
sys.path.append(os.getcwd())  # for windows
sys.path.append(expanduser('~/code/hesaff'))
import ibeis
from ibeis.dev import sysres
import utool
from ibeis.injest import injest_testdata
from vtool.tests import grabdata

workdir = ibeis.sysres.get_workdir()


TESTDB0 = join(workdir, 'testdb0')
TESTDB1 = join(workdir, 'testdb1')
TESTDB_GUIALL = join(workdir, 'testdb_guiall')


def delete_testdbs():
    utool.delete(TESTDB0)
    utool.delete(TESTDB1)
    utool.delete(TESTDB_GUIALL)


def make_testdb0():
    from ibeis.tests.test_gui_import_images import TEST_GUI_IMPORT_IMAGES
    from ibeis.tests.test_gui_add_roi import TEST_GUI_ADD_ROI
    main_locals = ibeis.main(dbdir=TESTDB0, gui=True)
    ibs = main_locals['ibs']
    back = main_locals['back']
    assert back is not None, str(main_locals)
    assert ibs is not None, str(main_locals)
    TEST_GUI_IMPORT_IMAGES(ibs, back)
    TEST_GUI_ADD_ROI(ibs, back)
    sysres.set_default_dbdir(TESTDB0)


def reset_testdbs():
    grabdata.ensure_testdata()
    delete_testdbs()
    make_testdb0()
    injest_testdata.injest_testdata()


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()  # For windows
    reset_testdbs()
