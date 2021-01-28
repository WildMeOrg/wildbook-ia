# -*- coding: utf-8 -*-
# See also conftest.py documentation at https://docs.pytest.org/en/stable/fixture.html#conftest-py-sharing-fixture-functions
"""This module is implicitly used by ``pytest`` to load testing configuration and fixtures."""
from pathlib import Path

import pytest
import sqlalchemy

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
    'testdb_assigner',
    # Not a populated database, but used by wbia.dbio.export_subset:merge_databases
    'testdb_dst',
)


@pytest.fixture
def enable_wildbook_signal():
    """This sets the ``ENABLE_WILDBOOK_SIGNAL`` to False"""
    # TODO (16-Jul-12020) Document ENABLE_WILDBOOK_SIGNAL
    # ??? what is ENABLE_WILDBOOK_SIGNAL used for?
    import wbia

    setattr(wbia, 'ENABLE_WILDBOOK_SIGNAL', False)


@pytest.fixture(scope='session', autouse=True)
def postgres_base_uri(request):
    """The base URI connection string to postgres.
    This should contain all necessary connection information except the database name.

    """
    uri = request.config.getoption('postgres_uri')
    if not uri:
        # Not set, return None; indicates the tests are not to use postgres
        return None

    # If the URI contains a database name, we need to remove it
    from sqlalchemy.engine.url import make_url, URL

    url = make_url(uri)
    url_kwargs = {
        'drivername': url.drivername,
        'username': url.username,
        'password': url.password,
        'host': url.host,
        'port': url.port,
        # Purposely remove database and query.
        # 'database': None,
        # 'query': None,
    }
    base_uri = str(URL.create(**url_kwargs))
    return base_uri


class MonkeyPatchedGetWbiaDbUri:
    """Creates a monkey patched version of ``wbia.init.sysres.get_wbia_db_uri``
    to set the testing URI.

    """

    def __init__(self, base_uri: str):
        self.base_uri = base_uri

    def __call__(self, db_dir: str):
        """The monkeypatch of ``wbia.init.sysres.get_wbia_db_uri``"""
        uri = None
        # Reminder, base_uri could be None if running tests under sqlite
        if self.base_uri:
            db_name = self.get_db_name_from_db_dir(Path(db_dir))
            uri = self.replace_uri_database(self.base_uri, db_name)
        return uri

    def get_db_name_from_db_dir(self, db_dir: Path):
        """Discover the database name from the given ``db_dir``"""
        from wbia.init.sysres import get_workdir

        db_dir = db_dir.resolve()  # just in case
        work_dir = Path(get_workdir()).resolve()

        # Can we discover the database name?
        # if not db_dir.is_relative_to(workdir):  # >= Python 3.9
        if not str(work_dir) in str(db_dir):
            raise ValueError(
                'Strange circumstances have lead us to a place of '
                f"incongruity where the '{db_dir}' is not within '{work_dir}'"
            )

        # lowercase because database names are case insensitive
        return db_dir.name.lower()

    def replace_uri_database(self, uri: str, db_name: str):
        """Replace the database name in the given ``uri`` with the given ``db_name``"""
        from sqlalchemy.engine.url import make_url

        url = make_url(uri)
        url = url._replace(database=db_name)

        return str(url)


@pytest.fixture(scope='session', autouse=True)
@pytest.mark.usefixtures('enable_wildbook_signal')
def set_up_db(request, postgres_base_uri):
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
        if postgres_base_uri:
            engine = sqlalchemy.create_engine(postgres_base_uri)
            engine.execution_options(isolation_level='AUTOCOMMIT').execute(
                f'DROP DATABASE IF EXISTS {dbname}'
            )
            engine.execution_options(isolation_level='AUTOCOMMIT').execute(
                f'CREATE DATABASE {dbname}'
            )
            engine.dispose()

    # Monkey patch the global URI getter
    from wbia.init import sysres

    setattr(sysres, 'get_wbia_db_uri', MonkeyPatchedGetWbiaDbUri(postgres_base_uri))

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
