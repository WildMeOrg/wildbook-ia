#!/usr/bin/env python2.7
# TODO: ADD COPYRIGHT TAG
from __future__ import absolute_import, division, print_function
from ibeis.dev import sysres
from ibeis.dbio import ingest_database
from os.path import join
import ibeis
import six
from itertools import cycle
import utool as ut

__test__ = False  # This is not a test


def get_testdata_dir(ensure=True, key='testdb1'):
    """
    Gets test img directory and downloads it if it doesn't exist
    """
    testdata_map = {
        'testdb1': 'https://dl.dropboxusercontent.com/s/of2s82ed4xf86m6/testdata.zip'
    }
    zipped_testdata_url = testdata_map[key]
    testdata_dir = ut.grab_zipped_url(zipped_testdata_url, ensure=ensure)
    return testdata_dir


def get_test_gpaths(ndata=None, names=None, **kwargs):
    # Read ndata from args or command line
    """ DEPRICATE """
    ndata_arg = ut.get_argval('--ndata', type_=int, default=None, help_='use --ndata to specify bigger data')
    if ndata_arg is not None:
        ndata = ndata_arg
    imgdir = get_testdata_dir(**kwargs)
    gpath_list = sorted(list(ut.list_images(imgdir, full=True, recursive=True)))
    # Get only the gpaths of certain names
    if names is not None:
        gpath_list = [gpath for gpath in gpath_list if
                      ut.basename_noext(gpath) in names]
    # Get a some number of test images
    if ndata is not None:
        gpath_cycle = cycle(gpath_list)
        if six.PY2:
            gpath_list  = [gpath_cycle.next() for _ in range(ndata)]
        else:
            gpath_list  = [next(gpath_cycle) for _ in range(ndata)]
    return gpath_list


def delete_testdbs():
    workdir = ibeis.sysres.get_workdir()
    TESTDB0 = join(workdir, 'testdb0')
    TESTDB1 = join(workdir, 'testdb1')
    TESTDB_GUIALL = join(workdir, 'testdb_guiall')
    ut.delete(TESTDB0, ignore_errors=False)
    ut.delete(TESTDB1, ignore_errors=False)
    ut.delete(TESTDB_GUIALL, ignore_errors=False)


def delete_larger_testdbs():
    workdir = ibeis.sysres.get_workdir()
    ut.delete(join(workdir, 'PZ_MTEST'))
    ut.delete(join(workdir, 'NAUT_test'))


def make_testdb0():
    workdir = ibeis.sysres.get_workdir()
    TESTDB0 = join(workdir, 'testdb0')
    main_locals = ibeis.main(dbdir=TESTDB0, gui=False, allow_newdir=True)
    ibs = main_locals['ibs']
    assert ibs is not None, str(main_locals)
    gpath_list = list(map(ut.unixpath, get_test_gpaths()))
    #print('[RESET] gpath_list=%r' % gpath_list)
    gid_list = ibs.add_images(gpath_list)  # NOQA
    valid_gids = ibs.get_valid_gids()
    valid_aids = ibs.get_valid_aids()
    try:
        assert len(valid_aids) == 0, 'there are more than 0 annotations in an empty database!'
    except Exception as ex:
        ut.printex(ex, key_list=['valid_aids'])
        raise
    gid_list = valid_gids[0:1]
    bbox_list = [(0, 0, 100, 100)]
    aid = ibs.add_annots(gid_list, bbox_list=bbox_list)[0]
    #print('[RESET] NEW RID=%r' % aid)
    aids = ibs.get_image_aids(gid_list)[0]
    try:
        assert aid in aids, ('bad annotation adder: aid = %r, aids = %r' % (aid, aids))
    except Exception as ex:
        ut.printex(ex, key_list=['aid', 'aids'])
        raise


def ensure_larger_testing_dbs():
    workdir = ibeis.sysres.get_workdir()
    if not ut.checkpath(join(workdir, 'PZ_MTEST')):
        ibeis.ensure_pz_mtest()
    if not ut.checkpath(join(workdir, 'NAUT_test')):
        ibeis.ensure_nauts()


def ensure_smaller_testingdbs():
    get_testdata_dir(True)
    print("\n\nMAKE TESTDB0\n\n")
    make_testdb0()
    print("\n\nMAKE TESTDB1\n\n")
    ingest_database.ingest_standard_database('testdb1')


def reset_testdbs():
    argdict = ut.parse_dict_from_argv(
        {
            'reset_all': False
        }
    )
    if argdict['reset_all']:
        delete_larger_testdbs()
    delete_testdbs()
    ensure_smaller_testingdbs()
    ensure_larger_testing_dbs()
    workdir = ibeis.sysres.get_workdir()
    TESTDB1 = join(workdir, 'testdb1')
    sysres.set_default_dbdir(TESTDB1)


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m ibeis.tests.reset_testdbs

    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.tests.reset_testdbs import *  # NOQA
        >>> # build test data
        >>> # execute function
        >>> result = reset_testdbs()
        >>> # verify results
        >>> print(result)
    """
    import multiprocessing
    multiprocessing.freeze_support()  # For windows
    #ibeis._preload()
    reset_testdbs()
