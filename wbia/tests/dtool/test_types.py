# -*- coding: utf-8 -*-
import uuid

import numpy as np
import pytest
from sqlalchemy.engine import create_engine
from sqlalchemy.sql import text, bindparam
from sqlalchemy.types import Float

from wbia.dtool.types import Dict, Integer, List, NDArray, Number, UUID


@pytest.fixture(autouse=True)
def db():
    engine = create_engine('sqlite:///:memory:', echo=False)
    with engine.connect() as conn:
        yield conn


def test_dict(db):
    # Create a table that uses the type
    db.execute(text('CREATE TABLE test(x DICT)'))

    # Insert a dict value into the table
    insert_value = {
        'a': 1,
        'b': 2.2,
        'c': [[1, 2, 3], [4, 5, 6]],
    }
    # Hint: https://docs.sqlalchemy.org/en/13/core/tutorial.html#specifying-bound-parameter-behaviors
    stmt = text('INSERT INTO test(x) VALUES (:x)')
    stmt = stmt.bindparams(bindparam('x', type_=Dict))
    db.execute(stmt, x=insert_value)

    # Query for the value
    stmt = text('select x from test')
    # Hint: https://docs.sqlalchemy.org/en/13/core/tutorial.html#specifying-result-column-behaviors
    stmt = stmt.columns(x=Dict)
    results = db.execute(stmt)
    selected_value = results.fetchone()[0]
    for k, v in selected_value.items():
        assert v == insert_value[k]


def test_list(db):
    # Create a table that uses the type
    db.execute(text('CREATE TABLE test(x LIST)'))

    # Insert a list of list value into the table
    insert_value = [[1, 2, 3], [4, 5, 6]]
    # Hint: https://docs.sqlalchemy.org/en/13/core/tutorial.html#specifying-bound-parameter-behaviors
    stmt = text('INSERT INTO test(x) VALUES (:x)')
    stmt = stmt.bindparams(bindparam('x', type_=List))
    db.execute(stmt, x=insert_value)

    # Query for the value
    stmt = text('select x from test')
    # Hint: https://docs.sqlalchemy.org/en/13/core/tutorial.html#specifying-result-column-behaviors
    stmt = stmt.columns(x=List)
    results = db.execute(stmt)
    selected_value = results.fetchone()[0]
    assert selected_value == insert_value


@pytest.mark.parametrize('num_type', (np.float32, np.float64))
def test_numpy_floats(db, num_type):
    # Verifies that a numpy float can be translated

    # Create a table that uses the type
    db.execute(text('CREATE TABLE test(x REAL)'))

    # Insert a uuid value into the table
    insert_value = num_type(8.0000008)
    # Hint: https://docs.sqlalchemy.org/en/13/core/tutorial.html#specifying-bound-parameter-behaviors
    stmt = text('INSERT INTO test(x) VALUES (:x)')
    stmt = stmt.bindparams(bindparam('x', type_=Float))
    db.execute(stmt, x=insert_value)

    # Query for the value
    stmt = text('SELECT x FROM test')
    # Hint: https://docs.sqlalchemy.org/en/13/core/tutorial.html#specifying-result-column-behaviors
    stmt = stmt.columns(x=Float)
    results = db.execute(stmt)
    selected_value = results.fetchone()[0]
    assert selected_value == insert_value


np_number_types = (
    np.int8,
    np.int16,
    np.int32,
    np.int64,
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    np.float32,
    np.float64,
)


@pytest.mark.parametrize('num_type', np_number_types)
def test_numpy_ints(db, num_type):
    # Create a table that uses the type
    db.execute(text('CREATE TABLE test(x INTEGER)'))

    # Insert a uuid value into the table
    insert_value = num_type(8)
    # Hint: https://docs.sqlalchemy.org/en/13/core/tutorial.html#specifying-bound-parameter-behaviors
    stmt = text('INSERT INTO test(x) VALUES (:x)')
    stmt = stmt.bindparams(bindparam('x', type_=Integer))
    db.execute(stmt, x=insert_value)

    # Query for the value
    stmt = text('SELECT x FROM test')
    # Hint: https://docs.sqlalchemy.org/en/13/core/tutorial.html#specifying-result-column-behaviors
    stmt = stmt.columns(x=Integer)
    results = db.execute(stmt)
    selected_value = results.fetchone()[0]
    assert selected_value == insert_value


def test_numpy_ndarray(db):
    # Create a table that uses the type
    db.execute(text('CREATE TABLE test(x NDARRAY)'))

    # Insert a numpy array value into the table
    insert_value = np.array([[1, 2, 3], [4, 5, 6]], np.int32)
    # Hint: https://docs.sqlalchemy.org/en/13/core/tutorial.html#specifying-bound-parameter-behaviors
    stmt = text('INSERT INTO test(x) VALUES (:x)')
    stmt = stmt.bindparams(bindparam('x', type_=NDArray))
    db.execute(stmt, x=insert_value)

    # Query for the value
    stmt = text('SELECT x FROM test')
    # Hint: https://docs.sqlalchemy.org/en/13/core/tutorial.html#specifying-result-column-behaviors
    stmt = stmt.columns(x=NDArray)
    results = db.execute(stmt)
    selected_value = results.fetchone()[0]
    assert (selected_value == insert_value).all()


np_numbers = (
    np.int8(120),
    np.int16(32767),
    np.int32(-2147483648),
    np.int64(9223372036854775807),
    np.uint8(255),
    np.uint16(65535),
    np.uint32(4294967295),
    np.uint64(18446744073709551615),
    np.float32(1.7976931348623157),
    np.float64(1.7976931348623157e308),
)


@pytest.mark.parametrize('num', np_numbers)
def test_numpy_sql_type(db, num):
    # Create a table that uses the type
    db.execute(text('CREATE TABLE test(x NUMPY)'))

    # Insert a numpy array value into the table
    insert_value = num
    # Hint: https://docs.sqlalchemy.org/en/13/core/tutorial.html#specifying-bound-parameter-behaviors
    stmt = text('INSERT INTO test(x) VALUES (:x)')
    stmt = stmt.bindparams(bindparam('x', type_=Number))
    db.execute(stmt, x=insert_value)

    # Query for the value
    stmt = text('SELECT x FROM test')
    # Hint: https://docs.sqlalchemy.org/en/13/core/tutorial.html#specifying-result-column-behaviors
    stmt = stmt.columns(x=Number)
    results = db.execute(stmt)
    selected_value = results.fetchone()[0]
    assert (selected_value == insert_value).all()


def test_uuid(db):
    # Create a table that uses the type
    db.execute(text('CREATE TABLE test(x UUID)'))

    # Insert a uuid value into the table
    insert_value = uuid.uuid4()
    # Hint: https://docs.sqlalchemy.org/en/13/core/tutorial.html#specifying-bound-parameter-behaviors
    stmt = text('INSERT INTO test(x) VALUES (:x)')
    stmt = stmt.bindparams(bindparam('x', type_=UUID))
    db.execute(stmt, x=insert_value)

    # Query for the value
    stmt = text('select x from test')
    # Hint: https://docs.sqlalchemy.org/en/13/core/tutorial.html#specifying-result-column-behaviors
    stmt = stmt.columns(x=UUID)
    results = db.execute(stmt)
    selected_value = results.fetchone()[0]
    assert selected_value == insert_value
