#!/usr/bin/env python2.7
"""
downloads standard test datasets. can delete them as well
"""
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


# Convert stanadardized names to true names
TEST_DBNAMES_MAP = {
    'nauts': 'NAUT_test',
    'mtest': 'PZ_MTEST',
    'testdb0': 'testdb0',
    'testdb1': 'testdb1',
    'testdb_guiall': 'testdb_guiall',
}


def delete_dbdir(dbname):
    ut.delete(join(ibeis.sysres.get_workdir(), dbname), ignore_errors=False)


def ensure_smaller_testingdbs():
    def make_testdb0():
        """ makes testdb0 """
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

    get_testdata_dir(True)
    if not ut.checkpath(join(ibeis.sysres.get_workdir(), 'testdb0'), verbose=True):
        print("\n\nMAKE TESTDB0\n\n")
        make_testdb0()
    if not ut.checkpath(join(ibeis.sysres.get_workdir(), 'testdb1'), verbose=True):
        print("\n\nMAKE TESTDB1\n\n")
        ingest_database.ingest_standard_database('testdb1')


def reset_testdbs():
    default_args = {'reset_' + key: False for key in six.iterkeys(TEST_DBNAMES_MAP) }
    default_args['reset_all'] = False
    argdict = ut.parse_dict_from_argv(default_args)
    if not any(list(six.itervalues(argdict))):
        # Default behavior is to reset the small dbs
        argdict['reset_testdb0'] = True
        argdict['reset_testdb1'] = True
        argdict['reset_testdb_guiall'] = True
    for key, dbname in six.iteritems(TEST_DBNAMES_MAP):
        if argdict.get('reset_' + key, False) or argdict['reset_all']:
            delete_dbdir(dbname)
    ensure_smaller_testingdbs()
    if not ut.checkpath(join(ibeis.sysres.get_workdir(), 'PZ_MTEST'), verbose=True):
        ibeis.ensure_pz_mtest()
    if not ut.checkpath(join(ibeis.sysres.get_workdir(), 'NAUT_test'), verbose=True):
        ibeis.ensure_nauts()
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
