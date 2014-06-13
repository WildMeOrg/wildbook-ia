#!/usr/bin/env python
# TODO: ADD COPYRIGHT TAG
from __future__ import absolute_import, division, print_function
from ibeis.dev import sysres
from ibeis.ingest import ingest_database
from os.path import join
from vtool.tests import grabdata
import ibeis
import utool

__test__ = False  # This is not a test


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
    main_locals = ibeis.main(dbdir=TESTDB0, gui=True, allow_newdir=True)
    ibs = main_locals['ibs']
    back = main_locals['back']
    assert back is not None, str(main_locals)
    assert ibs is not None, str(main_locals)
    TEST_GUI_IMPORT_IMAGES(ibs, back)
    TEST_GUI_ADD_ROI(ibs, back)


def reset_testdbs():
    grabdata.ensure_testdata()
    delete_testdbs()
    print("\n\nMAKE TESTDB0\n\n")
    make_testdb0()
    print("\n\nMAKE TESTDB1\n\n")
    ingest_database.ingest_standard_database('testdb1')
    sysres.set_default_dbdir(TESTDB1)


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()  # For windows
    reset_testdbs()
