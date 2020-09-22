# -*- coding: utf-8 -*-
import sqlite3
import uuid

import pytest

from wbia.dtool.sql_control import (
    METADATA_TABLE_COLUMNS,
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
