# -*- coding: utf-8 -*-
import sqlite3

import pytest

from wbia.dtool.dump import dump, dumps


@pytest.fixture
def db_conn():
    with sqlite3.connect(':memory:') as conn:
        conn.execute('create table metadata (id int primary key, x text)')
        conn.execute('create table foo (id int primary key, x text)')
        conn.execute('create table bar (id int primary key, x text)')
        for t in ('metadata', 'foo', 'bar'):
            for i in range(1, 3):
                conn.execute(f'insert into {t} (id, x) values (?, ?)', (i, f'_{i}_'))
        yield conn


def test_dumps(db_conn):
    # Write some stuff to dump
    data = dumps(db_conn)

    # Check schema is there
    assert 'CREATE TABLE bar' in data
    assert 'CREATE TABLE foo' in data
    assert 'CREATE TABLE metadata' in data

    # Check data inserts are there
    assert 'INSERT INTO "metadata" VALUES(1,\'_1_\');' in data


def test_dumps_schema_only(db_conn):
    # Write some stuff to dump
    data = dumps(db_conn, schema_only=True)

    # Check schema is there
    assert 'CREATE TABLE bar' in data
    assert 'CREATE TABLE foo' in data
    assert 'CREATE TABLE metadata' in data

    # Check data inserts are there
    assert 'INSERT INTO "foo" VALUES(1,\'_1_\');' not in data
    assert 'INSERT INTO "bar" VALUES(1,\'_1_\');' not in data
    # Check metadata inserts are there
    assert 'INSERT INTO "metadata" VALUES(1,\'_1_\');' in data


def test_dump(db_conn, tmp_path):
    fpath = tmp_path / 'dump.sql'
    with fpath.open('w') as fp:
        dump(db_conn, fp)

    with fpath.open('r') as fp:
        data = fp.read()

    # Check schema is there
    assert 'CREATE TABLE bar' in data
    assert 'CREATE TABLE foo' in data
    assert 'CREATE TABLE metadata' in data

    # Check data inserts are there
    assert 'INSERT INTO "foo" VALUES(1,\'_1_\');' in data
    assert 'INSERT INTO "bar" VALUES(1,\'_1_\');' in data
    # Check data inserts are there
    assert 'INSERT INTO "metadata" VALUES(1,\'_1_\');' in data
