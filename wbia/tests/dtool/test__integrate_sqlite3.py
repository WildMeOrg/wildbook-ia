# -*- coding: utf-8 -*-
import sqlite3
import uuid

import numpy as np
import pytest

# We do not explicitly call code in this module because
# importing the following module is execution of the code.
import wbia.dtool._integrate_sqlite3  # noqa


@pytest.fixture
def db():
    with sqlite3.connect(':memory:', detect_types=sqlite3.PARSE_DECLTYPES) as con:
        yield con


np_number_types = (
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
)


@pytest.mark.parametrize('num_type', np_number_types)
def test_register_numpy_dtypes_ints(db, num_type):
    # The magic takes place in the register_numpy_dtypes function,
    # which is implicitly called on module import.

    # Create a table that uses the type
    db.execute('create table test(x integer)')

    # Insert a uuid value into the table
    insert_value = num_type(8)
    db.execute('insert into test(x) values (?)', (insert_value,))
    # Query for the value
    cur = db.execute('select x from test')
    selected_value = cur.fetchone()[0]
    assert selected_value == insert_value


@pytest.mark.parametrize('num_type', (np.float32, np.float64))
def test_register_numpy_dtypes_floats(db, num_type):
    # The magic takes place in the register_numpy_dtypes function,
    # which is implicitly called on module import.

    # Create a table that uses the type
    db.execute('create table test(x real)')

    # Insert a uuid value into the table
    insert_value = num_type(8.0000008)
    db.execute('insert into test(x) values (?)', (insert_value,))
    # Query for the value
    cur = db.execute('select x from test')
    selected_value = cur.fetchone()[0]
    assert selected_value == insert_value


@pytest.mark.parametrize('type_name', ('numpy', 'ndarray'))
def test_register_numpy(db, type_name):
    # The magic takes place in the register_numpy function,
    # which is implicitly called on module import.

    # Create a table that uses the type
    db.execute(f'create table test(x {type_name})')

    # Insert a numpy array value into the table
    insert_value = np.array([[1, 2, 3], [4, 5, 6]], np.int32)
    db.execute('insert into test(x) values (?)', (insert_value,))
    # Query for the value
    cur = db.execute('select x from test')
    selected_value = cur.fetchone()[0]
    assert (selected_value == insert_value).all()


def test_register_uuid(db):
    # The magic takes place in the register_uuid function,
    # which is implicitly called on module import.

    # Create a table that uses the type
    db.execute('create table test(x uuid)')

    # Insert a uuid value into the table
    insert_value = uuid.uuid4()
    db.execute('insert into test(x) values (?)', (insert_value,))
    # Query for the value
    cur = db.execute('select x from test')
    selected_value = cur.fetchone()[0]
    assert selected_value == insert_value


def test_register_dict(db):
    # The magic takes place in the register_dict function,
    # which is implicitly called on module import.

    # Create a table that uses the type
    db.execute('create table test(x dict)')

    # Insert a dict value into the table
    insert_value = {
        'a': 1,
        'b': 2.2,
        'c': [[1, 2, 3], [4, 5, 6]],
    }
    db.execute('insert into test(x) values (?)', (insert_value,))
    # Query for the value
    cur = db.execute('select x from test')
    selected_value = cur.fetchone()[0]
    for k, v in selected_value.items():
        assert v == insert_value[k]


def test_register_list(db):
    # The magic takes place in the register_list function,
    # which is implicitly called on module import.

    # Create a table that uses the type
    db.execute('create table test(x list)')

    # Insert a list of list value into the table
    insert_value = [[1, 2, 3], [4, 5, 6]]
    db.execute('insert into test(x) values (?)', (insert_value,))
    # Query for the value
    cur = db.execute('select x from test')
    selected_value = cur.fetchone()[0]
    assert selected_value == insert_value
