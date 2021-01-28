# -*- coding: utf-8 -*-
"""
Definition of SQLAlchemy event listeners

See also, https://docs.sqlalchemy.org/en/latest/core/event.html

"""
from sqlalchemy import event
from sqlalchemy.schema import Table
from sqlalchemy.sql import text

from .types import SQL_TYPE_TO_SA_TYPE


# TODO (26-Sept-12020) Cache the results of this function.
def _discovery_table_columns(inspector, table_name):
    """Discover the original column type information in a _dialect_ specific way"""
    dialect = inspector.engine.dialect.name
    with inspector.engine.connect() as conn:
        if dialect == 'sqlite':
            # See also, https://sqlite.org/pragma.html#pragma_table_info
            result = conn.execute(f"PRAGMA TABLE_INFO('{table_name}')")
            #: column-id, name, data-type, nullable, default-value, is-primary-key
            info_rows = result.fetchall()
            names_to_types = {info[1]: info[2] for info in info_rows}
        elif dialect == 'postgresql':
            result = conn.execute(
                text(
                    """SELECT
                           row_number() over () - 1,
                           column_name,
                           coalesce(domain_name, data_type),
                           CASE WHEN is_nullable = 'YES' THEN 0 ELSE 1 END,
                           column_default,
                           column_name = (
                               SELECT column_name
                               FROM information_schema.table_constraints
                               NATURAL JOIN information_schema.constraint_column_usage
                               WHERE table_name = :table_name
                               AND constraint_type = 'PRIMARY KEY'
                               LIMIT 1
                           ) AS pk
                    FROM information_schema.columns
                    WHERE table_name = :table_name"""
                ),
                table_name=table_name,
            )
            info_rows = result.fetchall()
            names_to_types = {info[1]: info[2] for info in info_rows}
        else:
            raise RuntimeError(
                f"Unknown dialect ('{dialect}'), can't introspect column information."
            )
    return names_to_types


def _discover_specific_type(inspector, table_name, column_name):
    """Discover the specific type for a table's column.

    Args:
        inspector (Inspector): SQLAlchemy Inspector instance
        table_name (str): name of the table
        column_name (str): name of the column

    Returns:
        _ (str): type as defined in SQL

    """
    names_to_types = _discovery_table_columns(inspector, table_name)
    #: No need to check for existence, because we already know it's been found by SQLAlchemy
    return names_to_types[column_name]


@event.listens_for(Table, 'column_reflect')
def assign_user_defined_types_on_column_reflect(inspector, table, column_info):
    """Assigns our ``UserDefinedType``s on :class:`Table` reflection.

    See also, https://docs.sqlalchemy.org/en/latest/core/events.html#sqlalchemy.events.DDLEvents.column_reflect

    """
    # Unfortunately the `column_info` doesn't provide info about the original type text.
    data_type = _discover_specific_type(inspector, table.name, column_info['name'])
    try:
        # Map the SQL data-type to our SQLAlchemy user-defined type
        column_cls = SQL_TYPE_TO_SA_TYPE[data_type]
    except KeyError:
        pass  # Not one of our types
    else:
        # Assign the user-defined column type
        column_info['type'] = column_cls()
