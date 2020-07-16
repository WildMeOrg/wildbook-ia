#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
downloads standard test datasets. can delete them as well
"""
# TODO: ADD COPYRIGHT TAG
from __future__ import absolute_import, division, print_function, unicode_literals
from wbia.init import sysres
from wbia.dbio import ingest_database
from os.path import join
import wbia
import six
from itertools import cycle
import utool as ut

(print, rrr, profile) = ut.inject2(__name__)

__test__ = False  # This is not a test


def testdb2_stuff():
    """
    tar -zcvf testdb2.tar.gz testdb2/
    """
    import wbia

    ibs = wbia.opendb('testdb2')

    # ibs.ensure_contributor_rowids()

    gid_list = ibs.get_valid_gids()

    # Group gids by species
    image_species_list = ut.get_list_column(
        ibs.unflat_map(ibs.get_annot_species_rowids, ibs.get_image_aids(gid_list)), 0
    )

    new_contributor_rowid1 = ibs.add_new_temp_contributor(
        offset=len(ibs.get_valid_contributor_rowids())
    )
    new_contributor_rowid2 = ibs.add_new_temp_contributor(
        offset=len(ibs.get_valid_contributor_rowids())
    )

    gids1, gids2 = list(ut.group_items(gid_list, image_species_list).values())

    party_rowids = ibs.add_party(['TestCar1', 'TestCar2'])
    partyid1, partyid2 = party_rowids
    ibs.set_image_contributor_rowid(gids1, [new_contributor_rowid1] * len(gids1))
    ibs.set_image_contributor_rowid(gids2, [new_contributor_rowid2] * len(gids2))
    ibs.set_image_party_rowids(gids1, [partyid1] * len(gids1))
    ibs.set_image_party_rowids(gids2, [partyid2] * len(gids2))

    # image_contributor_rowid_list = ibs.get_image_contributor_rowid(gid_list)


def get_testdata_dir(ensure=True, key='testdb1'):
    """
    Gets test img directory and downloads it if it doesn't exist
    """
    testdata_map = {
        'testdb1': 'https://wildbookiarepository.azureedge.net/data/testdata.zip',
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
    'testdb2': 'testdb2',
    'testdb_guiall': 'testdb_guiall',
    'wds': 'wd_peter2',
}


def delete_dbdir(dbname):
    ut.delete(join(wbia.sysres.get_workdir(), dbname), ignore_errors=False)


def ensure_smaller_testingdbs():
    """
    Makes the smaller test databases
    """

    def make_testdb0():
        """ makes testdb0 """

        def get_test_gpaths(ndata=None, names=None, **kwargs):
            # Read ndata from args or command line
            """ DEPRICATE """
            ndata_arg = ut.get_argval(
                '--ndata',
                type_=int,
                default=None,
                help_='use --ndata to specify bigger data',
            )
            if ndata_arg is not None:
                ndata = ndata_arg
            imgdir = get_testdata_dir(**kwargs)
            gpath_list = sorted(list(ut.list_images(imgdir, full=True, recursive=True)))
            # Get only the gpaths of certain names
            if names is not None:
                gpath_list = [
                    gpath for gpath in gpath_list if ut.basename_noext(gpath) in names
                ]
            # Get a some number of test images
            if ndata is not None:
                gpath_cycle = cycle(gpath_list)
                if six.PY2:
                    gpath_list = [gpath_cycle.next() for _ in range(ndata)]
                else:
                    gpath_list = [next(gpath_cycle) for _ in range(ndata)]
            return gpath_list

        workdir = wbia.sysres.get_workdir()
        TESTDB0 = join(workdir, 'testdb0')
        main_locals = wbia.main(dbdir=TESTDB0, gui=False, allow_newdir=True)
        ibs = main_locals['ibs']
        assert ibs is not None, str(main_locals)
        gpath_list = list(map(ut.unixpath, get_test_gpaths()))
        # print('[RESET] gpath_list=%r' % gpath_list)
        gid_list = ibs.add_images(gpath_list)  # NOQA
        valid_gids = ibs.get_valid_gids()
        valid_aids = ibs.get_valid_aids()
        try:
            assert (
                len(valid_aids) == 0
            ), 'there are more than 0 annotations in an empty database!'
        except Exception as ex:
            ut.printex(ex, key_list=['valid_aids'])
            raise
        gid_list = valid_gids[0:1]
        bbox_list = [(0, 0, 100, 100)]
        aid = ibs.add_annots(gid_list, bbox_list=bbox_list)[0]
        # print('[RESET] NEW RID=%r' % aid)
        aids = ibs.get_image_aids(gid_list)[0]
        try:
            assert aid in aids, 'bad annotation adder: aid = %r, aids = %r' % (aid, aids,)
        except Exception as ex:
            ut.printex(ex, key_list=['aid', 'aids'])
            raise

    get_testdata_dir(True)
    if not ut.checkpath(join(wbia.sysres.get_workdir(), 'testdb0'), verbose=True):
        print('\n\nMAKE TESTDB0\n\n')
        make_testdb0()
    if not ut.checkpath(join(wbia.sysres.get_workdir(), 'testdb1'), verbose=True):
        print('\n\nMAKE TESTDB1\n\n')
        ingest_database.ingest_standard_database('testdb1')


def reset_testdbs(**kwargs):
    # Step 0) Parse Args
    wbia.ENABLE_WILDBOOK_SIGNAL = False
    default_args = {'reset_' + key: False for key in six.iterkeys(TEST_DBNAMES_MAP)}
    default_args['reset_all'] = False
    default_args.update(kwargs)
    argdict = ut.parse_dict_from_argv(default_args)
    if not any(list(six.itervalues(argdict))):
        # Default behavior is to reset the small dbs
        argdict['reset_testdb0'] = True
        argdict['reset_testdb1'] = True
        argdict['reset_testdb_guiall'] = True

    # Step 1) Delete DBs to be Reset
    for key, dbname in six.iteritems(TEST_DBNAMES_MAP):
        if argdict.get('reset_' + key, False) or argdict['reset_all']:
            delete_dbdir(dbname)

    # Step 3) Ensure DBs that dont exist
    ensure_smaller_testingdbs()
    workdir = sysres.get_workdir()
    if not ut.checkpath(join(workdir, 'PZ_MTEST'), verbose=True):
        wbia.ensure_pz_mtest()
    if not ut.checkpath(join(workdir, 'NAUT_test'), verbose=True):
        wbia.ensure_nauts()
    if not ut.checkpath(join(workdir, 'wd_peter2'), verbose=True):
        wbia.ensure_wilddogs()
    if not ut.checkpath(join(workdir, 'testdb2'), verbose=True):
        wbia.init.sysres.ensure_testdb2()

    # Step 4) testdb1 becomes the main database
    workdir = sysres.get_workdir()
    TESTDB1 = join(workdir, 'testdb1')
    sysres.set_default_dbdir(TESTDB1)


def reset_mtest():
    r"""
    CommandLine:
        python -m wbia --tf reset_mtest

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.tests.reset_testdbs import *  # NOQA
        >>> result = reset_mtest()
    """
    # Hack, this function does not have a utool main
    return reset_testdbs(reset_mtest=True)


if __name__ == '__main__':
    r"""
    CommandLine:
        python -m wbia.tests.reset_testdbs

    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.tests.reset_testdbs import *  # NOQA
        >>> result = reset_testdbs()
        >>> # verify results
        >>> print(result)
    """
    import multiprocessing

    multiprocessing.freeze_support()  # For windows
    # wbia._preload()
    reset_testdbs()
