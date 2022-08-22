# -*- coding: utf-8 -*-
# flake8: noqa

# BBB (7-Sept-12020)
# The following is meant to be used in place of `import sqlite3`
# because importing it from here will ensure our type integrations are loaded.
# See `_integrate_sqlite3` module for details.
import sqlite3
import sqlite3 as lite

import wbia.dtool.events
from wbia.dtool import base, depcache_control, depcache_table, sql_control
from wbia.dtool.base import *  # NOQA
from wbia.dtool.base import (
    AlgoResult,
    Config,
    MatchResult,
    VsManySimilarityRequest,
    VsOneSimilarityRequest,
)
from wbia.dtool.depcache_control import DependencyCache, make_depcache_decors
from wbia.dtool.depcache_table import ExternalStorageException, ExternType
from wbia.dtool.sql_control import SQLDatabaseController
from wbia.dtool.types import TYPE_TO_SQLTYPE
