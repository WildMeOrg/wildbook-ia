# -*- coding: utf-8 -*-
import sys

import pytest

from wbia import sysres
from wbia.init.sysres import get_args_dbdir


class TestGetArgsDbdir:
    # Monkeypatches functions used within get_args_dbdir to scope things.
    # Tests verify it passes through the correct function by marking the results.

    @pytest.fixture(autouse=True)
    def setup(self, monkeypatch):
        monkeypatch.setattr(sysres, 'db_to_dbdir', self.monkey_db_to_dbdir)
        monkeypatch.setattr(sysres, 'realpath', self.monkey_realpath)

    DB_MARK = '%'
    PATH_MARK = '!'

    def monkey_db_to_dbdir(self, db, **kwargs):
        # Just mark that we've seen it.
        # Ignorant of kwargs, noqa!
        return self.get_db_to_dbdir_marked(db)

    def monkey_realpath(self, path):
        # Just mark that we've seen it.
        return self.get_realpath_marked(path)

    def get_db_to_dbdir_marked(self, x):
        return f'{x}{self.DB_MARK}'

    def get_realpath_marked(self, x):
        return f'{x}{self.PATH_MARK}'

    def test_no_args(self):
        try:
            get_args_dbdir()
        except ValueError as exc:
            assert exc.args[0] == 'Must specify at least db, dbdir, or defaultdb'

    def test_db_arg(self):
        # The function first defaults to the specified function arguments.
        target = 'testdb1'
        d = get_args_dbdir(None, False, target, None)
        assert d == self.get_db_to_dbdir_marked(target)

    def test_dbdir_arg(self):
        # The function first defaults to the specified function arguments.
        target = 'foo'
        d = get_args_dbdir(None, False, None, target)
        assert d == self.get_realpath_marked(target)

    def test_cli_db(self, monkeypatch):
        target = 'foo'
        monkeypatch.setattr(sys, 'argv', ['--db', target])
        # ... then command line arguments are used.
        d = get_args_dbdir(None, False, None, None)
        assert d == self.get_db_to_dbdir_marked(target)

    def test_cli_dbdir(self, monkeypatch):
        target = 'foo'
        monkeypatch.setattr(sys, 'argv', ['--dbdir', target])
        # ... then command line arguments are used.
        d = get_args_dbdir(None, False, None, None)
        assert d == self.get_realpath_marked(target)

    def test_defaultdb(self):
        target = 'foo'
        # In all other circumstances defaultdb is used.
        d = get_args_dbdir(defaultdb=target)
        assert d == self.get_db_to_dbdir_marked(target)

    def test_defaultdb_cache(self, monkeypatch):
        # Monkeypatch the target function to guarantee the result.
        target = '<monkey>'
        monkeypatch.setattr(sysres, 'get_default_dbdir', lambda: target)

        # If defaultdb='cache' then the most recently used database directory is returned.
        d = get_args_dbdir(defaultdb='cache')
        assert d == target
