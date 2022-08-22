# -*- coding: utf-8 -*-
import pytest
from sqlalchemy import Column, Integer, MetaData, Table
from sqlalchemy.engine import create_engine

# On import the events will be registered with SQLAlchemy.
# Therefore, the testing target is whatever triggers registered the event listeners.
import wbia.dtool.events  # noqa
from wbia.dtool import types


@pytest.fixture(autouse=True)
def db():
    engine = create_engine('sqlite:///:memory:', echo=False)
    with engine.connect() as conn:
        yield conn


TYPES = [
    types.Dict,
    types.List,
    types.NDArray,
    types.Number,
    types.UUID,
]


def _make_comparable_Table_Columns(table):
    """Given a SQLAlchemy Table, iterate over the columns
    to make a comparable object with another table's columns

    """
    return {c.name: c.type.__class__ for c in table.columns}


def test_column_reflection(db):
    """Creates a table that uses all of our ``UserDefinedType``s defined in ``wbia.dtool.types``"""
    table_name = 'test_udt'
    creation_md = MetaData()
    creation_columns = [Column(chr(97 + i), type_) for i, type_ in enumerate(TYPES)]
    # Create the table
    defined_table = Table(
        table_name,
        creation_md,
        Column('id', Integer(), primary_key=True),
        *creation_columns
    )
    defined_table.create(db.engine)

    # Create a new metadata object to test reflection
    md = MetaData()
    # Just insure the new metadata object is empty
    assert len(md.tables) == 0

    # Call the target that calls the event code
    reflected_table = Table(table_name, md, autoload=True, autoload_with=db.engine)

    # Note, the id column is useful for testing the exception case in the event listener,
    # but we need not compare it. Also, it gets reflected as a slightly different type.
    def drop_id_column(columns):
        {k: v for k, v in columns.items() if k != 'id'}

    reflected_columns = drop_id_column(_make_comparable_Table_Columns(reflected_table))
    defined_columns = drop_id_column(_make_comparable_Table_Columns(defined_table))
    assert reflected_columns == defined_columns
