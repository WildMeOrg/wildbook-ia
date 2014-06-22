#!/usr/bin/env python2.7
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
    utool.delete(TESTDB0, ignore_errors=False)
    utool.delete(TESTDB1, ignore_errors=False)
    utool.delete(TESTDB_GUIALL, ignore_errors=False)


def make_testdb0():
    main_locals = ibeis.main(dbdir=TESTDB0, gui=False, allow_newdir=True)
    ibs = main_locals['ibs']
    assert ibs is not None, str(main_locals)
    gpath_list = map(utool.unixpath, grabdata.get_test_gpaths())
    print('[RESET] gpath_list=%r' % gpath_list)
    gid_list = ibs.add_images(gpath_list)  # NOQA
    valid_gids = ibs.get_valid_gids()
    valid_rids = ibs.get_valid_rids()
    try:
        assert len(valid_rids) == 0, 'there are more than 0 rois in an empty database!'
    except Exception as ex:
        utool.printex(ex, key_list=['valid_rids'])
        raise
    gid_list = valid_gids[0:1]
    bbox_list = [(0, 0, 100, 100)]
    rid = ibs.add_rois(gid_list, bbox_list=bbox_list)[0]
    print('[RESET] NEW RID=%r' % rid)
    rids = ibs.get_image_rids(gid_list)[0]
    try:
        assert rid in rids, ('bad roi adder: rid = %r, rids = %r' % (rid, rids))
    except Exception as ex:
        utool.printex(ex, key_list=['rid', 'rids'])
        raise


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
