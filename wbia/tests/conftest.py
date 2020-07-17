# -*- coding: utf-8 -*-
# See also conftest.py documentation at https://docs.pytest.org/en/stable/fixture.html#conftest-py-sharing-fixture-functions
"""This module is implicitly used by ``pytest`` to load testing configuration and fixtures."""
from pathlib import Path

import pytest

from wbia.init.sysres import (
    ensure_nauts,
    ensure_pz_mtest,
    ensure_testdb2,
    ensure_wilddogs,
    get_workdir,
    set_default_dbdir,
)
from .reset_testdbs import (
    TEST_DBNAMES_MAP,
    delete_dbdir,
    ensure_smaller_testingdbs,
)


@pytest.fixture
def enable_wildbook_signal():
    """This sets the ``ENABLE_WILDBOOK_SIGNAL`` to False"""
    # TODO (16-Jul-12020) Document ENABLE_WILDBOOK_SIGNAL
    # ??? what is ENABLE_WILDBOOK_SIGNAL used for?
    import wbia

    setattr(wbia, 'ENABLE_WILDBOOK_SIGNAL', False)


@pytest.fixture(scope='session', autouse=True)
@pytest.mark.usefixtures('enable_wildbook_signal')
def set_up_db():
    """
    Sets up the testing databases.
    This fixture is set to run automatically any any test run of wbia.

    """
    # Delete DBs, if they exist
    # FIXME (16-Jul-12020) this fixture does not cleanup after itself to preserve exiting usage behavior
    for dbname in TEST_DBNAMES_MAP.values():
        delete_dbdir(dbname)

    # Set up DBs
    ensure_smaller_testingdbs()
    ensure_pz_mtest()
    ensure_nauts()
    ensure_wilddogs()
    ensure_testdb2()

    # Set testdb1 as the main database
    workdir = Path(get_workdir())
    default_db_dir = workdir / 'testdb1'
    # FIXME (16-Jul-12020) Set this only for the test session
    set_default_dbdir(default_db_dir.resolve())
