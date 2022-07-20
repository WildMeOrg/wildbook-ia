# -*- coding: utf-8 -*-
import uuid
from functools import partial

import numpy as np
import pytest
import sqlalchemy.exc
from sqlalchemy import MetaData, Table
from sqlalchemy.sql import select, text

from wbia.dtool.sql_control import (
    METADATA_TABLE_COLUMNS,
    SQLDatabaseController,
)


@pytest.fixture
def ctrlr():
    return SQLDatabaseController('sqlite:///:memory:', 'testing')


def make_table_definition(name, depends_on=[]):
    """Creates a table definition for use with the controller's add_table method"""
    definition = {
        'tablename': name,
        'coldef_list': [
            (f'{name}_id', 'INTEGER PRIMARY KEY'),
            ('meta_labeler_id', 'INTEGER NOT NULL'),
            ('indexer_id', 'INTEGER NOT NULL'),
            ('config_id', 'INTEGER DEFAULT 0'),
            ('data', 'TEXT'),
        ],
        'docstr': f'docstr for {name}',
        'superkeys': [
            ('meta_labeler_id', 'indexer_id', 'config_id'),
        ],
        'dependson': depends_on,
    }
    return definition


def test_instantiation_with_table_reflection(tmp_path):
    db_file = (tmp_path / 'testing.db').resolve()
    creating_ctrlr = SQLDatabaseController(f'sqlite:///{db_file}', 'testing')
    # Assumes `add_table` is functional. If you run into failing problems
    # check for failures around this method first.
    created_tables = []
    table_names = map(
        'table_{}'.format,
        (
            'a',
            'b',
            'c',
        ),
    )
    for t in table_names:
        creating_ctrlr.add_table(**make_table_definition(t, depends_on=created_tables))
        # Build up of denpendence
        created_tables.append(t)

    # Delete the controller
    del creating_ctrlr

    # Create the controller again for reflection (testing target)
    ctrlr = SQLDatabaseController(f'sqlite:///{db_file}', 'testing')
    # Verify the tables are loaded on instantiation
    assert list(ctrlr._sa_metadata.tables.keys()) == ['metadata'] + created_tables
    # Note, we don't have to check for the contents of the tables,
    # because that's machinery within SQLAlchemy,
    # which will have been tested by SQLAlchemy.


class TestSchemaModifiers:
    """Testing the API that creates, modifies or deletes schema elements"""

    @pytest.fixture(autouse=True)
    def fixture(self, ctrlr):
        self.ctrlr = ctrlr

    make_table_definition = staticmethod(make_table_definition)

    @property
    def _table_factory(self):
        return partial(Table, autoload=True, autoload_with=self.ctrlr._engine)

    def reflect_table(self, name, metadata=None):
        """Using SQLAlchemy to reflect the table at the given ``name``
        to return a SQLAlchemy Table object

        """
        if metadata is None:
            metadata = MetaData()
        return self._table_factory(name, metadata)

    def test_make_add_table_sqlstr(self):
        table_definition = self.make_table_definition(
            'foobars', depends_on=['meta_labelers', 'indexers']
        )

        # Call the target
        sql = self.ctrlr._make_add_table_sqlstr(**table_definition)

        expected = (
            'CREATE TABLE IF NOT EXISTS foobars ( '
            'foobars_id INTEGER PRIMARY KEY, '
            'meta_labeler_id INTEGER NOT NULL, '
            'indexer_id INTEGER NOT NULL, '
            'config_id INTEGER DEFAULT 0, '
            'data TEXT, '
            'CONSTRAINT unique_foobars_meta_labeler_id_indexer_id_config_id '
            'UNIQUE (meta_labeler_id, indexer_id, config_id) )'
        )
        assert sql.text == expected

    def test_add_table(self):
        # Two tables...
        # .. used in the creation of bars table
        foos_definition = self.make_table_definition('foos')
        # .. bars table depends on foos table
        bars_definition = self.make_table_definition('bars', depends_on=['foos'])
        # We test against bars table and basically neglect foos table

        # Call the target
        self.ctrlr.add_table(**foos_definition)
        self.ctrlr.add_table(**bars_definition)

        # Check the table has been added and verify details
        # Use sqlalchemy's reflection
        md = MetaData()
        bars = self.reflect_table('bars', md)
        metadata = self.reflect_table('metadata', md)

        # Check the table's column definitions
        expected_bars_columns = [
            ('bars_id', 'wbia.dtool.types.Integer'),
            ('config_id', 'wbia.dtool.types.Integer'),
            ('data', 'sqlalchemy.sql.sqltypes.TEXT'),
            ('indexer_id', 'wbia.dtool.types.Integer'),
            ('meta_labeler_id', 'wbia.dtool.types.Integer'),
        ]
        found_bars_columns = [
            (c.name, '.'.join([c.type.__class__.__module__, c.type.__class__.__name__]))
            for c in bars.columns
        ]
        assert sorted(found_bars_columns) == expected_bars_columns
        # Check the table's constraints
        expected_constraint_info = [
            ('PrimaryKeyConstraint', None, ['bars_id']),
            (
                'UniqueConstraint',
                'unique_bars_meta_labeler_id_indexer_id_config_id',
                ['meta_labeler_id', 'indexer_id', 'config_id'],
            ),
        ]
        found_constraint_info = [
            (x.__class__.__name__, x.name, [c.name for c in x.columns])
            for x in bars.constraints
        ]
        assert sorted(found_constraint_info) == expected_constraint_info

        # Check for metadata entries
        results = self.ctrlr._engine.execute(
            select([metadata.c.metadata_key, metadata.c.metadata_value]).where(
                metadata.c.metadata_key.like('bars_%')
            )
        )
        expected_metadata_rows = [
            ('bars_docstr', 'docstr for bars'),
            ('bars_superkeys', "[('meta_labeler_id', 'indexer_id', 'config_id')]"),
            ('bars_dependson', "['foos']"),
        ]
        assert results.fetchall() == expected_metadata_rows

    def test_rename_table(self):
        # Assumes `add_table` passes to reduce this test's complexity.
        table_name = 'cookies'
        self.ctrlr.add_table(**self.make_table_definition(table_name))

        # Call the target
        new_table_name = 'deserts'
        self.ctrlr.rename_table(table_name, new_table_name)

        # Check the table has been renamed use sqlalchemy's reflection
        md = MetaData()
        metadata = self.reflect_table('metadata', md)

        # Reflecting the table is enough to check that it's been renamed.
        self.reflect_table(new_table_name, md)

        # Check for metadata entries have been renamed.
        results = self.ctrlr._engine.execute(
            select([metadata.c.metadata_key, metadata.c.metadata_value]).where(
                metadata.c.metadata_key.like(f'{new_table_name}_%')
            )
        )
        expected_metadata_rows = [
            (f'{new_table_name}_docstr', f'docstr for {table_name}'),
            (
                f'{new_table_name}_superkeys',
                "[('meta_labeler_id', 'indexer_id', 'config_id')]",
            ),
            (f'{new_table_name}_dependson', '[]'),
        ]
        assert results.fetchall() == expected_metadata_rows

    def test_drop_table(self):
        # Assumes `add_table` passes to reduce this test's complexity.
        table_name = 'cookies'
        self.ctrlr.add_table(**self.make_table_definition(table_name))

        # Call the target
        self.ctrlr.drop_table(table_name)

        # Check the table using sqlalchemy's reflection
        md = MetaData()
        metadata = self.reflect_table('metadata', md)

        # This error in the attempt to reflect indicates the table has been removed.
        with pytest.raises(sqlalchemy.exc.NoSuchTableError):
            self.reflect_table(table_name, md)

        # Check for metadata entries have been renamed.
        results = self.ctrlr._engine.execute(
            select([metadata.c.metadata_key, metadata.c.metadata_value]).where(
                metadata.c.metadata_key.like(f'{table_name}_%')
            )
        )
        assert results.fetchall() == []

    def test_drop_all_table(self):
        # Assumes `add_table` passes to reduce this test's complexity.
        table_names = ['cookies', 'pies', 'cakes']
        for name in table_names:
            self.ctrlr.add_table(**self.make_table_definition(name))

        # Call the target
        self.ctrlr.drop_all_tables()

        # Check the table using sqlalchemy's reflection
        md = MetaData()
        metadata = self.reflect_table('metadata', md)

        # This error in the attempt to reflect indicates the table has been removed.
        for name in table_names:
            with pytest.raises(sqlalchemy.exc.NoSuchTableError):
                self.reflect_table(name, md)

        # Check for the absents of metadata for the removed tables.
        results = self.ctrlr._engine.execute(select([metadata.c.metadata_key]))
        expected_metadata_rows = [
            ('database_init_uuid',),
            ('database_version',),
            ('metadata_docstr',),
            ('metadata_superkeys',),
        ]
        assert results.fetchall() == expected_metadata_rows


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
        insert_stmt = text(
            'INSERT INTO metadata (metadata_key, metadata_value) VALUES (:key, :value)'
        )
        for key, value in self.data.items():
            unprefixed_name = key.split('_')[-1]
            if METADATA_TABLE_COLUMNS[unprefixed_name]['is_coded_data']:
                value = repr(value)
            self.ctrlr._engine.execute(insert_stmt, key=key, value=value)

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
        assert not self.ctrlr._engine.execute(
            f"SELECT * FROM metadata WHERE metadata_key = 'foo_{key}'"
        ).fetchone()

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
        assert not self.ctrlr._engine.execute(
            f"SELECT * FROM metadata WHERE metadata_key = 'foo_{key}'"
        ).fetchone()

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
        assert not self.ctrlr._engine.execute(
            f"SELECT * FROM metadata WHERE metadata_key = 'foo_{key}'"
        ).fetchone()

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


class BaseAPITestCase:
    """Testing the primary *usage* API"""

    @pytest.fixture(autouse=True)
    def fixture(self, ctrlr):
        self.ctrlr = ctrlr

    def make_table(self, name):
        self.ctrlr._engine.execute(
            f'CREATE TABLE IF NOT EXISTS {name} '
            '(id INTEGER PRIMARY KEY, x TEXT, y INTEGER, z REAL)'
        )

    def populate_table(self, name):
        """To be used in conjunction with ``make_table`` to populate the table
        with records from 0 to 9.

        """
        insert_stmt = text(f'INSERT INTO {name} (x, y, z) VALUES (:x, :y, :z)')
        for i in range(0, 10):
            x, y, z = (
                (i % 2) and 'odd' or 'even',
                i,
                i * 2.01,
            )
            self.ctrlr._engine.execute(insert_stmt, x=x, y=y, z=z)


class TestExecutionAPI(BaseAPITestCase):
    def test_executeone(self):
        table_name = 'test_executeone'
        self.make_table(table_name)

        # Create some dummy records
        self.populate_table(table_name)

        # Call the testing target
        result = self.ctrlr.executeone(text(f'SELECT id, y FROM {table_name}'))

        assert result == [(i + 1, i) for i in range(0, 10)]

    def test_executeone_using_fetchone_behavior(self):
        table_name = 'test_executeone'
        self.make_table(table_name)

        # Call the testing target with `fetchone` method's returning behavior.
        result = self.ctrlr.executeone(
            text(f'SELECT id, y FROM {table_name}'), use_fetchone_behavior=True
        )

        # IMO returning None is correct,
        # because that's the expectation from `fetchone`'s DBAPI spec.
        assert result is None

    def test_executeone_without_results(self):
        table_name = 'test_executeone'
        self.make_table(table_name)

        # Call the testing target
        result = self.ctrlr.executeone(text(f'SELECT id, y FROM {table_name}'))

        # IMO returning None is correct,
        # because that's the expectation from `fetchone`'s DBAPI spec.
        assert result == []

    def test_executeone_on_insert(self):
        # Should return id after an insert
        table_name = 'test_executeone'
        self.make_table(table_name)

        # Create some dummy records
        self.populate_table(table_name)

        # Call the testing target
        result = self.ctrlr.executeone(
            text(f'INSERT INTO {table_name} (y) VALUES (:y)'), {'y': 10}
        )

        # Cursory check that the result is a single int value
        assert result == [11]  # the result list with one unwrapped value

        # Check for the actual value associated with the resulting id
        inserted_value = self.ctrlr._engine.execute(
            text(f'SELECT id, y FROM {table_name} WHERE rowid = :rowid'),
            rowid=result[0],
        ).fetchone()
        assert inserted_value == (
            11,
            10,
        )

    def test_executemany(self):
        table_name = 'test_executemany'
        self.make_table(table_name)

        # Create some dummy records
        self.populate_table(table_name)

        # Call the testing target
        results = self.ctrlr.executemany(
            text(f'SELECT id, y FROM {table_name} where x = :x'),
            ({'x': 'even'}, {'x': 'odd'}),
            unpack_scalars=False,
        )

        # Check for results
        evens = [(i + 1, i) for i in range(0, 10) if not i % 2]
        odds = [(i + 1, i) for i in range(0, 10) if i % 2]
        assert results == [evens, odds]

    def test_executemany_transaction(self):
        table_name = 'test_executemany'
        self.make_table(table_name)

        # Test a failure to execute in the transaction to test the transaction boundary.
        insert = text(f'INSERT INTO {table_name} (x, y, z) VALUES (:x, y:, :z)')
        params = [
            dict(x='even', y=0, z=0.0),
            dict(x='odd', y=1, z=1.01),
            dict(x='oops', z=2.02),  # error
            dict(x='odd', y=3, z=3.03),
        ]
        with pytest.raises(sqlalchemy.exc.OperationalError):
            # Call the testing target
            results = self.ctrlr.executemany(insert, params)

        # Check for results
        results = self.ctrlr._engine.execute(f'select count(*) from {table_name}')
        assert results.fetchone()[0] == 0

    def test_executeone_for_single_column(self):
        # Should unwrap the resulting query value (no tuple wrapping)
        table_name = 'test_executeone'
        self.make_table(table_name)

        # Create some dummy records
        self.populate_table(table_name)

        # Call the testing target
        result = self.ctrlr.executeone(text(f'SELECT y FROM {table_name}'))

        # Note the unwrapped values, rather than [(i,) ...]
        assert result == [i for i in range(0, 10)]


class TestAdditionAPI(BaseAPITestCase):
    def test_add(self):
        table_name = 'test_add'
        self.make_table(table_name)

        parameter_values = []
        for i in range(0, 10):
            x, y, z = (
                (i % 2) and 'odd' or 'even',
                i,
                i * 2.01,
            )
            parameter_values.append((x, y, z))

        # Call the testing target
        ids = self.ctrlr._add(table_name, ['x', 'y', 'z'], parameter_values)

        # Verify the resulting ids
        assert ids == [i + 1 for i in range(0, len(parameter_values))]
        # Verify addition of records
        results = self.ctrlr._engine.execute(f'SELECT id, x, y, z FROM {table_name}')
        expected = [(i + 1, x, y, z) for i, (x, y, z) in enumerate(parameter_values)]
        assert results.fetchall() == expected


class TestGettingAPI(BaseAPITestCase):
    def test_get_where_without_where_condition(self):
        table_name = 'test_get_where'
        self.make_table(table_name)

        # Create some dummy records
        self.populate_table(table_name)

        # Call the testing target
        results = self.ctrlr.get_where(
            table_name,
            ['id', 'y'],
            tuple(),
            None,
        )

        # Verify query
        assert results == [(i + 1, i) for i in range(0, 10)]

    def test_scalar_get_where(self):
        table_name = 'test_get_where'
        self.make_table(table_name)

        # Create some dummy records
        self.populate_table(table_name)

        # Call the testing target
        results = self.ctrlr.get_where(
            table_name,
            ['id', 'y'],
            ({'id': 1},),
            'id = :id',
        )
        evens = results[0]

        # Verify query
        assert evens == (1, 0)

    def test_multi_row_get_where(self):
        table_name = 'test_get_where'
        self.make_table(table_name)

        # Create some dummy records
        self.populate_table(table_name)

        # Call the testing target
        results = self.ctrlr.get_where(
            table_name,
            ['id', 'y'],
            ({'x': 'even'}, {'x': 'odd'}),
            'x = :x',
            unpack_scalars=False,  # this makes it more than one row of results
        )
        evens = results[0]
        odds = results[1]

        # Verify query
        assert evens == [(i + 1, i) for i in range(0, 10) if not i % 2]
        assert odds == [(i + 1, i) for i in range(0, 10) if i % 2]

    def test_get_where_eq(self):
        table_name = 'test_get_where_eq'
        self.make_table(table_name)

        # Create some dummy records
        self.populate_table(table_name)

        # Call the testing target
        results = self.ctrlr.get_where_eq(
            table_name,
            ['id', 'y'],
            (['even', 8], ['odd', 7]),  # params_iter
            ('x', 'y'),  # where_colnames
            op='AND',
            unpack_scalars=True,
        )

        # Verify query
        assert results == [(9, 8), (8, 7)]

    def test_get_all(self):
        # Make a table for records
        table_name = 'test_getting'
        self.make_table(table_name)

        # Create some dummy records
        insert_stmt = text(f'INSERT INTO {table_name} (x, y, z) VALUES (:x, :y, :z)')
        with self.ctrlr.connect() as conn:
            for i in range(0, 10):
                x, y, z = (str(i), i, i * 2.01)
                conn.execute(insert_stmt, x=x, y=y, z=z)

            # Build the expect results of the testing target
            results = conn.execute(f'SELECT id, x, z FROM {table_name}')
            rows = results.fetchall()
            row_mapping = {row[0]: row[1:] for row in rows if row[1]}

        # Call the testing target
        data = self.ctrlr.get(table_name, ['x', 'z'])

        # Verify getting
        assert data == [r for r in row_mapping.values()]

    def test_get_by_id(self):
        # Make a table for records
        table_name = 'test_getting'
        self.make_table(table_name)

        # Create some dummy records
        insert_stmt = text(f'INSERT INTO {table_name} (x, y, z) VALUES (:x, :y, :z)')
        with self.ctrlr.connect() as conn:
            for i in range(0, 10):
                x, y, z = (str(i), i, i * 2.01)
                conn.execute(insert_stmt, x=x, y=y, z=z)

        # Call the testing target
        requested_ids = [2, 4, 6]
        data = self.ctrlr.get(table_name, ['x', 'z'], requested_ids)

        # Build the expect results of the testing target
        sql_array = ', '.join([str(id) for id in requested_ids])
        with self.ctrlr.connect() as conn:
            results = conn.execute(
                f'SELECT x, z FROM {table_name} WHERE id in ({sql_array})'
            )
            expected = results.fetchall()
        # Verify getting
        assert data == expected

    def test_get_by_numpy_array_of_ids(self):
        # Make a table for records
        table_name = 'test_getting'
        self.make_table(table_name)

        # Create some dummy records
        insert_stmt = text(f'INSERT INTO {table_name} (x, y, z) VALUES (:x, :y, :z)')
        with self.ctrlr.connect() as conn:
            for i in range(0, 10):
                x, y, z = (str(i), i, i * 2.01)
                conn.execute(insert_stmt, x=x, y=y, z=z)

        # Call the testing target
        requested_ids = np.array([2, 4, 6])
        data = self.ctrlr.get(table_name, ['x', 'z'], requested_ids)

        # Build the expect results of the testing target
        sql_array = ', '.join([str(id) for id in requested_ids])
        with self.ctrlr.connect() as conn:
            results = conn.execute(
                f'SELECT x, z FROM {table_name} WHERE id in ({sql_array})'
            )
            expected = results.fetchall()
        # Verify getting
        assert data == expected

    def test_get_as_unique(self):
        # This test could be inaccurate, because this logical path appears
        # to be bolted on the side. Usage of this path's feature is unknown.

        # Make a table for records
        table_name = 'test_getting'
        self.make_table(table_name)

        # Create some dummy records
        insert_stmt = text(f'INSERT INTO {table_name} (x, y, z) VALUES (:x, :y, :z)')
        with self.ctrlr.connect() as conn:
            for i in range(0, 10):
                x, y, z = (str(i), i, i * 2.01)
                conn.execute(insert_stmt, x=x, y=y, z=z)

        # Call the testing target
        # The table has a INTEGER PRIMARY KEY, which essentially maps to the rowid
        # in SQLite. So, we need not change the default `id_colname` param.
        requested_ids = [2, 4, 6]
        data = self.ctrlr.get(table_name, ['x'], requested_ids, assume_unique=True)

        # Build the expect results of the testing target
        sql_array = ', '.join([str(id) for id in requested_ids])
        with self.ctrlr.connect() as conn:
            results = conn.execute(
                f'SELECT x FROM {table_name} WHERE id in ({sql_array})'
            )
            # ... recall that the controller unpacks single values
            expected = [row[0] for row in results]
        # Verify getting
        assert data == expected


class TestSettingAPI(BaseAPITestCase):
    def test_setting(self):
        # Note, this is not a comprehensive test. It only attempts to test the SQL logic.
        # Make a table for records
        table_name = 'test_setting'
        self.make_table(table_name)

        # Create some dummy records
        insert_stmt = text(f'INSERT INTO {table_name} (x, y, z) VALUES (:x, :y, :z)')
        with self.ctrlr.connect() as conn:
            for i in range(0, 10):
                x, y, z = (str(i), i, i * 2.01)
                conn.execute(insert_stmt, x=x, y=y, z=z)

            results = conn.execute(f'SELECT id, CAST((y%2) AS BOOL) FROM {table_name}')
            rows = results.fetchall()
        ids = [row[0] for row in rows if row[1]]

        # Call the testing target
        self.ctrlr.set(
            table_name, ['x', 'z'], [('even', 0.0)] * len(ids), ids, id_colname='id'
        )

        # Verify setting
        sql_array = ', '.join([str(id) for id in ids])
        with self.ctrlr.connect() as conn:
            results = conn.execute(
                f'SELECT id, x, z FROM {table_name} ' f'WHERE id in ({sql_array})'
            )
            expected = sorted(map(lambda a: tuple([a] + ['even', 0.0]), ids))
            set_rows = sorted(results)
        assert set_rows == expected


class TestDeletionAPI(BaseAPITestCase):
    def test_delete(self):
        # Make a table for records
        table_name = 'test_delete'
        self.make_table(table_name)

        # Create some dummy records
        insert_stmt = text(f'INSERT INTO {table_name} (x, y, z) VALUES (:x, :y, :z)')
        with self.ctrlr.connect() as conn:
            for i in range(0, 10):
                x, y, z = (str(i), i, i * 2.01)
                conn.execute(insert_stmt, x=x, y=y, z=z)

            results = conn.execute(f'SELECT id, CAST((y % 2) AS BOOL) FROM {table_name}')
            rows = results.fetchall()
        del_ids = [row[0] for row in rows if row[1]]
        remaining_ids = sorted([row[0] for row in rows if not row[1]])

        # Call the testing target
        self.ctrlr.delete(table_name, del_ids, 'id')

        # Verify the deletion
        with self.ctrlr.connect() as conn:
            results = conn.execute(f'SELECT id FROM {table_name}')
            assert sorted([r[0] for r in results]) == remaining_ids

    def test_delete_rowid(self):
        # Make a table for records
        table_name = 'test_delete_rowid'
        self.make_table(table_name)

        # Create some dummy records
        insert_stmt = text(f'INSERT INTO {table_name} (x, y, z) VALUES (:x, :y, :z)')
        with self.ctrlr.connect() as conn:
            for i in range(0, 10):
                x, y, z = (str(i), i, i * 2.01)
                conn.execute(insert_stmt, x=x, y=y, z=z)

            results = conn.execute(
                f'SELECT rowid, CAST((y % 2) AS BOOL) FROM {table_name}'
            )
            rows = results.fetchall()
        del_ids = [row[0] for row in rows if row[1]]
        remaining_ids = sorted([row[0] for row in rows if not row[1]])

        # Call the testing target
        self.ctrlr.delete_rowids(table_name, del_ids)

        # Verify the deletion
        with self.ctrlr.connect() as conn:
            results = conn.execute(f'SELECT rowid FROM {table_name}')
            assert sorted([r[0] for r in results]) == remaining_ids
