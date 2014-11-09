#!/usr/bin/env python2.7
# TODO: ADD COPYRIGHT TAG
from __future__ import absolute_import, division, print_function
from ibeis.dev import sysres
from ibeis.dbio import ingest_database
from os.path import join
from vtool.tests import grabdata
import ibeis
import utool

__test__ = False  # This is not a test


def delete_testdbs():
    workdir = ibeis.sysres.get_workdir()
    TESTDB0 = join(workdir, 'testdb0')
    TESTDB1 = join(workdir, 'testdb1')
    TESTDB_GUIALL = join(workdir, 'testdb_guiall')
    utool.delete(TESTDB0, ignore_errors=False)
    utool.delete(TESTDB1, ignore_errors=False)
    utool.delete(TESTDB_GUIALL, ignore_errors=False)


def make_testdb0():
    workdir = ibeis.sysres.get_workdir()
    TESTDB0 = join(workdir, 'testdb0')
    main_locals = ibeis.main(dbdir=TESTDB0, gui=False, allow_newdir=True)
    ibs = main_locals['ibs']
    assert ibs is not None, str(main_locals)
    gpath_list = list(map(utool.unixpath, grabdata.get_test_gpaths()))
    #print('[RESET] gpath_list=%r' % gpath_list)
    gid_list = ibs.add_images(gpath_list)  # NOQA
    valid_gids = ibs.get_valid_gids()
    valid_aids = ibs.get_valid_aids()
    try:
        assert len(valid_aids) == 0, 'there are more than 0 annotations in an empty database!'
    except Exception as ex:
        utool.printex(ex, key_list=['valid_aids'])
        raise
    gid_list = valid_gids[0:1]
    bbox_list = [(0, 0, 100, 100)]
    aid = ibs.add_annots(gid_list, bbox_list=bbox_list)[0]
    #print('[RESET] NEW RID=%r' % aid)
    aids = ibs.get_image_aids(gid_list)[0]
    try:
        assert aid in aids, ('bad annotation adder: aid = %r, aids = %r' % (aid, aids))
    except Exception as ex:
        utool.printex(ex, key_list=['aid', 'aids'])
        raise


def reset_testdbs():
    grabdata.ensure_testdata()
    delete_testdbs()
    print("\n\nMAKE TESTDB0\n\n")
    make_testdb0()
    print("\n\nMAKE TESTDB1\n\n")
    ingest_database.ingest_standard_database('testdb1')
    workdir = ibeis.sysres.get_workdir()
    TESTDB1 = join(workdir, 'testdb1')
    sysres.set_default_dbdir(TESTDB1)


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()  # For windows
    #ibeis._preload()
    reset_testdbs()
