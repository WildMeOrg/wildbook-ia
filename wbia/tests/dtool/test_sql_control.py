# -*- coding: utf-8 -*-
import sqlite3

import pytest

from wbia.dtool.sql_control import (
    TIMEOUT,
    SQLDatabaseController,
)


@pytest.fixture
def ctrlr():
    return SQLDatabaseController.from_uri(':memory:')


def test_instantiation(ctrlr):
    # Check for basic connection information
    assert ctrlr.uri == ':memory:'
    assert ctrlr.timeout == TIMEOUT

    # Check for a connection, that would have been made during instantiation
    assert isinstance(ctrlr.connection, sqlite3.Connection)
    assert ctrlr.cur is None


def test_version(ctrlr):
    v = ctrlr.get_db_version()
    assert v == '0.0.0'
