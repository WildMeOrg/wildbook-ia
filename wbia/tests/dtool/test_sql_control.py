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
    assert isinstance(ctrlr.cur, sqlite3.Cursor)


def test_safely_get_db_version(ctrlr):
    v = ctrlr.get_db_version(ensure=True)
    assert v == '0.0.0'


def test_unsafely_get_db_version(ctrlr):
    v = ctrlr.get_db_version(ensure=False)
    assert v == '0.0.0'


class TestMakeAddTableSql:
    def test_success(self, ctrlr):
        table_name = 'keypoints'
        column_definitions = [
            ('keypoint_rowid', 'INTEGER PRIMARY KEY'),
            ('chip_rowid', 'INTEGER NOT NULL'),
            ('config_rowid', 'INTEGER DEFAULT 0'),
            ('kpts', 'NDARRAY'),
            ('num', 'INTEGER'),
        ]
        sql = ctrlr._make_add_table_sqlstr(table_name, column_definitions)

        expected = (
            'CREATE TABLE IF NOT EXISTS keypoints '
            '( keypoint_rowid INTEGER PRIMARY KEY, '
            'chip_rowid INTEGER NOT NULL, '
            'config_rowid INTEGER DEFAULT 0, '
            'kpts NDARRAY, num INTEGER )'
        )
        assert sql == expected

    def test_no_column_definition(self, ctrlr):
        table_name = 'keypoints'
        column_definitions = []
        with pytest.raises(ValueError) as caught_exc:
            ctrlr._make_add_table_sqlstr(table_name, column_definitions)
        assert 'empty coldef_list' in caught_exc.value.args[0]

    def test_with_superkeys(self, ctrlr):
        table_name = 'keypoints'
        column_definitions = [
            ('keypoint_rowid', 'INTEGER PRIMARY KEY'),
            ('chip_rowid', 'INTEGER NOT NULL'),
            ('config_rowid', 'INTEGER DEFAULT 0'),
            ('kpts', 'NDARRAY'),
            ('num', 'INTEGER'),
        ]
        superkeys = [('kpts', 'num'), ('config_rowid',)]
        sql = ctrlr._make_add_table_sqlstr(
            table_name, column_definitions, superkeys=superkeys
        )

        expected = (
            'CREATE TABLE IF NOT EXISTS keypoints '
            '( keypoint_rowid INTEGER PRIMARY KEY, '
            'chip_rowid INTEGER NOT NULL, '
            'config_rowid INTEGER DEFAULT 0, '
            'kpts NDARRAY, num INTEGER, '
            'CONSTRAINT superkey UNIQUE (kpts,num), '
            'CONSTRAINT superkey UNIQUE (config_rowid) )'
        )
        assert sql == expected
