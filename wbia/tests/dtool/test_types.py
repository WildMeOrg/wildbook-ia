# -*- coding: utf-8 -*-
import uuid

import pytest
from sqlalchemy.engine import create_engine
from sqlalchemy.sql import text, bindparam

from wbia.dtool.types import Dict, List, UUID


@pytest.fixture(autouse=True)
def db():
    engine = create_engine('sqlite:///:memory:', echo=False,)
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
