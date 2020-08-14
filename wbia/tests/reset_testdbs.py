#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
downloads standard test datasets. can delete them as well
"""
# TODO: ADD COPYRIGHT TAG
from itertools import cycle
from os.path import join

import six
import utool as ut

import wbia
from wbia.dbio import ingest_database
from wbia.init import sysres
from wbia.tests.helpers import get_testdata_dir


(print, rrr, profile) = ut.inject2(__name__)

__test__ = False  # This is not a test


# Convert stanadardized names to true names
TEST_DBNAMES_MAP = {
    'nauts': 'NAUT_test',
    'mtest': 'PZ_MTEST',
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
    get_testdata_dir(ensure=True)
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
        >>> # FIXME failing-test (22-Jul-2020) sqlite3.OperationalError: attempt to write a readonly database
        >>> # xdoctest: +SKIP
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
