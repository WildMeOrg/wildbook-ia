# -*- coding: utf-8 -*-
"""Operations for dumping a database to file"""
__all__ = (
    'dump',
    'dumps',
)


def _iter_dump(db_connection, schema_only=False):
    """\
    Generator function that serializes the database associated with
    ``db_connection`` as formatted ``str``.

    Args:
        db_connection (sqlite3.Connection): database instance to dump
        schema_only (bool): flag to only dump the schema

    Returns:
        str: serialized string of database

    """
    for line in db_connection.iterdump():
        # ??? (23-Jul-12020) This seems a bit dicey, good idea?
        #     This may only work in sqlite environment.
        if schema_only and line.startswith('INSERT'):
            # FIXME (23-Jul-12020) hardcoded table name, use METADATA_TABLE
            #       when circular import is resolved.
            # if metadata is requested then allow those inserts
            if 'INSERT INTO "metadata' not in line:
                continue
        yield f'{line}\n'


def dumps(*args, **kwargs):
    """\
    Serialize the database associated with ``db_connection`` as formatted ``str``.

    Args:
        db_connection (sqlite3.Connection): database instance to dump
        schema_only (bool): flag to only dump the schema

    Returns:
        str: serialized string of database

    """

    return ''.join([s for s in _iter_dump(*args, **kwargs)])


def dump(db_connection, fp, **kwargs):
    """\
    Serialize the database associated with ``db_connection`` as a formatted
    stream to ``fp`` (a ``.write()``-supporting file-like object).

    This is a side-effect only function that writes to ``fp``.

    Args:
        db_connection (sqlite3.Connection): database instance to dump
        fp (file-like object): output stream
        schema_only (bool): flag to only dump the schema

    Returns:
        None

    """
    for line in _iter_dump(db_connection, **kwargs):
        fp.write(line)
