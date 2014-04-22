#!/usr/bin/env python
# TODO: ADD COPYRIGHT TAG
from __future__ import absolute_import, division, print_function
import sys
import os
sys.path.append(os.getcwd())  # for windows
import ibeis
from ibeis.dev import main_commands
import utool
from os.path import join

workdir = ibeis.params.get_workdir()
TESTDB_DIR = join(workdir, 'testdb')


def delete_testdbs():
    utool.delete(TESTDB_DIR)


def make_testdbs():
    from test_gui_import_images import TEST_GUI_IMPORT_IMAGES
    from test_gui_add_roi import TEST_GUI_ADD_ROI
    dbdir = TESTDB_DIR
    main_locals = ibeis.main(dbdir=dbdir, gui=True)
    ibs = main_locals['ibs']
    back = main_locals['back']
    TEST_GUI_IMPORT_IMAGES(ibs, back)
    TEST_GUI_ADD_ROI(ibs, back)
    main_commands.set_default_dbdir(TESTDB_DIR)


def reset_testdbs():
    delete_testdbs()
    make_testdbs()


if __name__ == '__main__':
    reset_testdbs()
