# -*- coding: utf-8 -*-
# flake8: noqa
from __future__ import absolute_import, division, print_function, unicode_literals

# The following is meant to be used in place of `import sqlite3`
# because importing it from here will ensure our type integrations are loaded.
# See `_integrate_sqlite3` module for details.
import sqlite3

from wbia.dtool import _integrate_sqlite3 as lite
from wbia.dtool import base
from wbia.dtool import sql_control
from wbia.dtool import depcache_control
from wbia.dtool import depcache_table

from wbia.dtool.depcache_control import DependencyCache, make_depcache_decors
from wbia.dtool.base import (
    AlgoResult,
    MatchResult,
    Config,
    VsManySimilarityRequest,
    VsOneSimilarityRequest,
)
from wbia.dtool.depcache_table import ExternalStorageException, ExternType
from wbia.dtool.base import *  # NOQA
from wbia.dtool.sql_control import SQLDatabaseController
from wbia.dtool.types import TYPE_TO_SQLTYPE
