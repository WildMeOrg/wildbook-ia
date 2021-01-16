# -*- coding: utf-8 -*-
def pytest_addoption(parser):
    # This needs to be in the project root not in wbia/conftest.py otherwise
    # e.g. "pytest --gui" doesn't work.
    parser.addoption('--fixme', action='store_true')
    parser.addoption('--gui', action='store_true')
    parser.addoption('--show', action='store_true')
    parser.addoption('--slow', action='store_true')
    parser.addoption('--tomcat', action='store_true')
    parser.addoption('--web-tests', action='store_true')
    parser.addoption('--weird', action='store_true')
    parser.addoption(
        '--disable-refresh-db',
        action='store_true',
        help=(
            'disables the set_up_db fixture from rebuilding the db, '
            "instead it will reuse the previous test run's db"
        ),
    )
    parser.addoption(
        '--with-postgres-uri',
        dest='postgres_uri',
        help=(
            'used to enable tests to run against a Postgres database '
            '(note, the uri should use a superuser role)'
        ),
    )
