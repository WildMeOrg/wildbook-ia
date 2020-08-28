# -*- coding: utf-8 -*-
# See also conftest.py documentation at https://docs.pytest.org/en/stable/fixture.html#conftest-py-sharing-fixture-functions
"""This module is implicitly used by ``pytest`` to load testing configuration and fixtures."""
import os
from functools import wraps
from pathlib import Path

import pytest

from wbia.dbio import ingest_database
from wbia.init.sysres import (
    delete_dbdir,
    ensure_nauts,
    ensure_pz_mtest,
    ensure_testdb2,
    ensure_wilddogs,
    get_workdir,
    set_default_dbdir,
)


TEST_DBNAMES = (
    'NAUT_test',
    'PZ_MTEST',
    'testdb1',
    'testdb2',
    'testdb_guiall',
    'wd_peter2',
)


# Global marker for determining the availablity of postgres
# set by db_uri fixture and used by requires_postgresql decorator
_POSTGRES_AVAILABLE = None


#
# Decorators
#


def requires_postgresql(func):
    """Test decorator to mark a test that requires postgresql

    Usage:

        @requires_postgresql
        def test_postgres_thing():
            # testing logic that requires postgres...
            assert True

    """
    # Firstly, skip if psycopg2 is not installed
    try:
        import psycopg2  # noqa:
    except ImportError:
        pytest.mark.skip('psycopg2 is not installed')(func)

    @wraps(func)
    def wrapper(*args, **kwargs):
        # We'll only know if we can connect to postgres during execution.
        if not _POSTGRES_AVAILABLE:  # see db_uri fixture for value definition
            pytest.skip('requires a postgresql connection URI')
        return func(*args, **kwargs)

    return wrapper


#
#  Fixtures
#


@pytest.fixture(scope='session', autouse=True)
def db_uri():
    """The DB URI to use with the tests.
    This value comes from ``WBIA_TESTING_BASE_DB_URI``.
    """
    # TODO (28-Aug-12020) Should we depend on the user supplying this value?
    #      Perhaps not at this level? Fail if not specified?
    uri = os.getenv('WBIA_TESTING_BASE_DB_URI', '')

    # Set postgres availablity marker
    global _POSTGRES_AVAILABLE
    _POSTGRES_AVAILABLE = uri.startswith('postgres')

    return uri


@pytest.fixture
def enable_wildbook_signal():
    """This sets the ``ENABLE_WILDBOOK_SIGNAL`` to False"""
    # TODO (16-Jul-12020) Document ENABLE_WILDBOOK_SIGNAL
    # ??? what is ENABLE_WILDBOOK_SIGNAL used for?
    import wbia

    setattr(wbia, 'ENABLE_WILDBOOK_SIGNAL', False)


@pytest.fixture(scope='session', autouse=True)
@pytest.mark.usefixtures('enable_wildbook_signal')
def set_up_db(request):
    """
    Sets up the testing databases.
    This fixture is set to run automatically any any test run of wbia.

    """
    # If selected, disable running the main logic of this fixture
    if request.config.getoption('--disable-refresh-db', False):
        # Assume the user knows what they are requesting
        return  # bale out

    # Delete DBs, if they exist
    # FIXME (16-Jul-12020) this fixture does not cleanup after itself to preserve exiting usage behavior
    for dbname in TEST_DBNAMES:
        delete_dbdir(dbname)

    # Set up DBs
    ingest_database.ingest_standard_database('testdb1')
    ensure_pz_mtest()
    ensure_nauts()
    ensure_wilddogs()
    ensure_testdb2()

    # Set testdb1 as the main database
    workdir = Path(get_workdir())
    default_db_dir = workdir / 'testdb1'
    # FIXME (16-Jul-12020) Set this only for the test session
    set_default_dbdir(default_db_dir.resolve())


@pytest.fixture(scope='session')
def xdoctest_namespace(set_up_db):
    """
    Inject names into the xdoctest namespace.
    """
    return dict()
