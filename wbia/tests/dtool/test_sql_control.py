# -*- coding: utf-8 -*-
import uuid

import pytest
from sqlalchemy.engine import Connection
from sqlalchemy.sql import text

from wbia.dtool.sql_control import (
    METADATA_TABLE_COLUMNS,
    TIMEOUT,
    SQLDatabaseController,
)


@pytest.fixture
def ctrlr():
    return SQLDatabaseController.from_uri('sqlite:///:memory:')


def test_instantiation(ctrlr):
    # Check for basic connection information
    assert ctrlr.uri == 'sqlite:///:memory:'
    assert ctrlr.timeout == TIMEOUT

    # Check for a connection, that would have been made during instantiation
    assert isinstance(ctrlr.connection, Connection)
    assert not ctrlr.connection.closed


def test_safely_get_db_version(ctrlr):
    v = ctrlr.get_db_version(ensure=True)
    assert v == '0.0.0'


def test_unsafely_get_db_version(ctrlr):
    v = ctrlr.get_db_version(ensure=False)
    assert v == '0.0.0'


class TestMetadataProperty:

    data = {
        'foo_docstr': 'lalala',
        'foo_relates': ['bar'],
    }

    @pytest.fixture(autouse=True)
    def fixture(self, ctrlr, monkeypatch):
        self.ctrlr = ctrlr
        # Allow our 'foo' table to fictitiously exist
        monkeypatch.setattr(self.ctrlr, 'get_table_names', self.monkey_get_table_names)

        # Create the metadata table
        self.ctrlr._ensure_metadata_table()

        # Create metadata in the table
        for key, value in self.data.items():
            unprefixed_name = key.split('_')[-1]
            if METADATA_TABLE_COLUMNS[unprefixed_name]['is_coded_data']:
                value = repr(value)
            self.ctrlr.executeone(
                'INSERT INTO metadata (metadata_key, metadata_value) VALUES (?, ?)',
                (key, value),
            )

    def monkey_get_table_names(self, *args, **kwargs):
        return ['foo', 'metadata']

    # ###
    # Test attribute access methods
    # ###

    def test_getter(self):
        # Check getting of a value by key
        assert self.data['foo_relates'] == self.ctrlr.metadata.foo.relates

        # ... docstr is an exception where other values have repr() used on them
        assert self.data['foo_docstr'] == self.ctrlr.metadata.foo.docstr

    def test_getting_unset_value(self):
        # Check getting an attribute with without a SQL record
        assert self.ctrlr.metadata.foo.superkeys is None

    def test_setter(self):
        # Check setting of a value by key
        key = 'shortname'
        value = 'fu'
        setattr(self.ctrlr.metadata.foo, key, value)

        new_value = getattr(self.ctrlr.metadata.foo, key)
        assert new_value == value

        # Check setting of a value by key, of list type
        key = 'superkeys'
        value = [
            ('a',),
            ('b', 'c'),
        ]
        setattr(self.ctrlr.metadata.foo, key, value)

        new_value = getattr(self.ctrlr.metadata.foo, key)
        assert new_value == value

        # ... docstr is an exception where other values have eval() used on them
        key = 'docstr'
        value = 'rarara'
        setattr(self.ctrlr.metadata.foo, key, value)

        new_value = getattr(self.ctrlr.metadata.foo, key)
        assert new_value == value

    def test_setting_to_none(self):
        # Check setting a value to None, essentially deleting it.
        key = 'shortname'
        value = None
        setattr(self.ctrlr.metadata.foo, key, value)

        new_value = getattr(self.ctrlr.metadata.foo, key)
        assert new_value == value

        # Also check the table does not have the record
        assert not self.ctrlr.executeone(
            f"SELECT * FROM metadata WHERE metadata_key = 'foo_{key}'"
        )

    def test_setting_unknown_key(self):
        # Check setting of an unknown metadata key
        key = 'smoo'
        value = 'thing'
        with pytest.raises(AttributeError):
            setattr(self.ctrlr.metadata.foo, key, value)

    def test_deleter(self):
        key = 'docstr'
        # Check we have the initial value set
        assert self.ctrlr.metadata.foo.docstr == self.data[f'foo_{key}']

        delattr(self.ctrlr.metadata.foo, key)
        # You can't really delete the attribute, but it does null the value.
        assert self.ctrlr.metadata.foo.docstr is None

        # Also check the table does not have the record
        assert not self.ctrlr.executeone(
            f"SELECT * FROM metadata WHERE metadata_key = 'foo_{key}'"
        )

    def test_database_attributes(self):
        # Check the database version
        assert self.ctrlr.metadata.database.version == '0.0.0'
        # Check the database init_uuid is a valid uuid.UUID
        assert isinstance(self.ctrlr.metadata.database.init_uuid, uuid.UUID)

    # ###
    # Test batch manipulation methods
    # ###

    def test_update(self):
        targets = {
            'superkeys': [('k',), ('x', 'y')],
            'dependson': ['baz'],
            'docstr': 'hahaha',
        }
        # Updating from dict
        self.ctrlr.metadata.foo.update(**targets)

        # Check for updates
        for k in targets:
            assert getattr(self.ctrlr.metadata.foo, k) == targets[k]

    # ###
    # Test item access methods proxy to attribute access methods
    # ###

    def test_getitem(self):
        # Check getting a value through the mapping interface
        assert self.ctrlr.metadata['foo'].table_name == 'foo'

    def test_getitem_for_unknown_table(self):
        # Check getting the metdata for an unknown table
        with pytest.raises(KeyError):
            assert self.ctrlr.metadata['bar']

    # ###
    # Test item access methods on TableMetada proxy to attribute access methods
    # ###

    def test_getitem_for_table_metadata(self):
        # Check getting a value through the mapping interface
        assert self.ctrlr.metadata['foo']['docstr'] == self.data['foo_docstr']
        assert list([k for k in self.ctrlr.metadata['foo']]) == list(
            METADATA_TABLE_COLUMNS.keys()
        )
        assert len([k for k in self.ctrlr.metadata['foo']]) == len(METADATA_TABLE_COLUMNS)

        # ... and for the database
        assert self.ctrlr.metadata['database']['version'] == '0.0.0'
        assert list(self.ctrlr.metadata['database'].keys()) == ['version', 'init_uuid']
        assert len(self.ctrlr.metadata['database']) == 2

    def test_setitem(self):
        # Check setting of a value by key
        key = 'shortname'
        value = 'fu'
        self.ctrlr.metadata.foo[key] = value

        new_value = getattr(self.ctrlr.metadata.foo, key)
        assert new_value == value

        # Check setting of a value by key, of list type
        key = 'superkeys'
        value = [
            ('a',),
            ('b', 'c'),
        ]
        self.ctrlr.metadata.foo[key] = value

        new_value = getattr(self.ctrlr.metadata.foo, key)
        assert new_value == value

        # ... docstr is an exception where other values have eval() used on them
        key = 'docstr'
        value = 'rarara'
        self.ctrlr.metadata.foo[key] = value

        new_value = getattr(self.ctrlr.metadata.foo, key)
        assert new_value == value

        # ... database version
        key = 'version'
        value = '1.1.1'
        self.ctrlr.metadata.database[key] = value

        new_value = getattr(self.ctrlr.metadata.database, key)
        assert new_value == value

    def test_delitem_for_table(self):
        key = 'docstr'
        # Check we have the initial value set
        self.ctrlr.metadata.foo[key] = 'not tobe'

        # You can't really delete the attribute, but it does null the value.
        del self.ctrlr.metadata.foo[key]
        assert self.ctrlr.metadata.foo.docstr is None

        # Also check the table does not have the record
        assert not self.ctrlr.executeone(
            f"SELECT * FROM metadata WHERE metadata_key = 'foo_{key}'"
        )

    def test_delitem_for_database(self):
        # You cannot delete database version metadata
        with pytest.raises(RuntimeError):
            del self.ctrlr.metadata.database['version']
        with pytest.raises(ValueError):
            self.ctrlr.metadata.database['version'] = None
        assert self.ctrlr.metadata.database.version == '0.0.0'

        # You cannot delete database init_uuid metadata
        with pytest.raises(RuntimeError):
            del self.ctrlr.metadata.database['init_uuid']
        with pytest.raises(ValueError):
            self.ctrlr.metadata.database['init_uuid'] = None
        # Check the value is still a uuid.UUID
        assert isinstance(self.ctrlr.metadata.database.init_uuid, uuid.UUID)


class TestAPI:
    """Testing the primary *usage* API"""

    @pytest.fixture(autouse=True)
    def fixture(self, ctrlr):
        self.ctrlr = ctrlr

    def make_table(self, name):
        self.ctrlr.connection.execute(
            f'CREATE TABLE IF NOT EXISTS {name} '
            '(id INTEGER PRIMARY KEY, x TEXT, y INTEGER, z REAL)'
        )

    def test_get_where_without_where_condition(self):
        table_name = 'test_get_where'
        self.make_table(table_name)

        # Create some dummy records
        insert_stmt = text(f'INSERT INTO {table_name} (x, y, z) VALUES (:x, :y, :z)')
        for i in range(0, 10):
            x, y, z = (
                (i % 2) and 'odd' or 'even',
                i,
                i * 2.01,
            )
            self.ctrlr.connection.execute(insert_stmt, x=x, y=y, z=z)

        # Call the testing target
        results = self.ctrlr.get_where(table_name, ['id', 'y'], tuple(), None,)

        # Verify query
        assert results == [(i + 1, i) for i in range(0, 10)]

    def test_scalar_get_where(self):
        table_name = 'test_get_where'
        self.make_table(table_name)

        # Create some dummy records
        insert_stmt = text(f'INSERT INTO {table_name} (x, y, z) VALUES (:x, :y, :z)')
        for i in range(0, 10):
            x, y, z = (
                (i % 2) and 'odd' or 'even',
                i,
                i * 2.01,
            )
            self.ctrlr.connection.execute(insert_stmt, x=x, y=y, z=z)

        # Call the testing target
        results = self.ctrlr.get_where(table_name, ['id', 'y'], ([1]), 'id = ?',)
        evens = results[0]

        # Verify query
        assert evens == (1, 0)

    def test_multi_row_get_where(self):
        table_name = 'test_get_where'
        self.make_table(table_name)

        # Create some dummy records
        insert_stmt = text(f'INSERT INTO {table_name} (x, y, z) VALUES (:x, :y, :z)')
        for i in range(0, 10):
            x, y, z = (
                (i % 2) and 'odd' or 'even',
                i,
                i * 2.01,
            )
            self.ctrlr.connection.execute(insert_stmt, x=x, y=y, z=z)

        # Call the testing target
        results = self.ctrlr.get_where(
            table_name,
            ['id', 'y'],
            (['even'], ['odd']),
            'x = ?',
            unpack_scalars=False,  # this makes it more than one row of results
        )
        evens = results[0]
        odds = results[1]

        # Verify query
        assert evens == [(i + 1, i) for i in range(0, 10) if not i % 2]
        assert odds == [(i + 1, i) for i in range(0, 10) if i % 2]

    def test_setting(self):
        # Note, this is not a comprehensive test. It only attempts to test the SQL logic.
        # Make a table for records
        table_name = 'test_setting'
        self.make_table(table_name)

        # Create some dummy records
        insert_stmt = text(f'INSERT INTO {table_name} (x, y, z) VALUES (:x, :y, :z)')
        for i in range(0, 10):
            x, y, z = (str(i), i, i * 2.01)
            self.ctrlr.connection.execute(insert_stmt, x=x, y=y, z=z)

        results = self.ctrlr.connection.execute(
            f'SELECT id, CAST((y%2) AS BOOL) FROM {table_name}'
        )
        rows = results.fetchall()
        ids = [row[0] for row in rows if row[1]]

        # Call the testing target
        self.ctrlr.set(
            table_name, ['x', 'z'], [('even', 0.0)] * len(ids), ids, id_colname='id'
        )

        # Verify setting
        sql_array = ', '.join([str(id) for id in ids])
        results = self.ctrlr.connection.execute(
            f'SELECT id, x, z FROM {table_name} ' f'WHERE id in ({sql_array})'
        )
        expected = sorted(map(lambda a: tuple([a] + ['even', 0.0]), ids))
        set_rows = sorted(results)
        assert set_rows == expected

    def test_delete(self):
        # Make a table for records
        table_name = 'test_delete'
        self.make_table(table_name)

        # Create some dummy records
        insert_stmt = text(f'INSERT INTO {table_name} (x, y, z) VALUES (:x, :y, :z)')
        for i in range(0, 10):
            x, y, z = (str(i), i, i * 2.01)
            self.ctrlr.connection.execute(insert_stmt, x=x, y=y, z=z)

        results = self.ctrlr.connection.execute(
            f'SELECT id, CAST((y % 2) AS BOOL) FROM {table_name}'
        )
        rows = results.fetchall()
        del_ids = [row[0] for row in rows if row[1]]
        remaining_ids = sorted([row[0] for row in rows if not row[1]])

        # Call the testing target
        self.ctrlr.delete(table_name, del_ids, 'id')

        # Verify the deletion
        results = self.ctrlr.connection.execute(f'SELECT id FROM {table_name}')
        assert sorted([r[0] for r in results]) == remaining_ids

    def test_delete_rowid(self):
        # Make a table for records
        table_name = 'test_delete_rowid'
        self.make_table(table_name)

        # Create some dummy records
        insert_stmt = text(f'INSERT INTO {table_name} (x, y, z) VALUES (:x, :y, :z)')
        for i in range(0, 10):
            x, y, z = (str(i), i, i * 2.01)
            self.ctrlr.connection.execute(insert_stmt, x=x, y=y, z=z)

        results = self.ctrlr.connection.execute(
            f'SELECT rowid, CAST((y % 2) AS BOOL) FROM {table_name}'
        )
        rows = results.fetchall()
        del_ids = [row[0] for row in rows if row[1]]
        remaining_ids = sorted([row[0] for row in rows if not row[1]])

        # Call the testing target
        self.ctrlr.delete_rowids(table_name, del_ids)

        # Verify the deletion
        results = self.ctrlr.connection.execute(f'SELECT rowid FROM {table_name}')
        assert sorted([r[0] for r in results]) == remaining_ids
