# -*- coding: utf-8 -*-
"""
Module Licence and docstring

TODO: ideally the ibeis.constants module would not be used here
and each function would use its own constant variables that are suffixed
with the last version number that they existed in

CommandLine:
    python -m ibeis.control.STAGING_SCHEMA --test-autogen_staging_schema
"""
from __future__ import absolute_import, division, print_function
from ibeis import constants as const

try:
    from ibeis.control import STAGING_SCHEMA_CURRENT
    UPDATE_CURRENT  = STAGING_SCHEMA_CURRENT.update_current
    VERSION_CURRENT = STAGING_SCHEMA_CURRENT.VERSION_CURRENT
except:
    UPDATE_CURRENT  = None
    VERSION_CURRENT = None
    print("[dbcache] NO STAGING_SCHEMA_CURRENT AUTO-GENERATED!")
import utool as ut
profile = ut.profile


REVIEW_ROWID        = 'review_rowid'


# =======================
# Schema Version 1.0.0
# =======================


@profile
def update_1_0_0(db, ibs=None):
    db.add_table(const.REVIEW_TABLE, (
        ('review_rowid',                 'INTEGER PRIMARY KEY'),
        ('annot_1_rowid',                'INTEGER NOT NULL'),
        ('annot_2_rowid',                'INTEGER NOT NULL'),
        ('review_count',                 'INTEGER NOT NULL'),
        ('review_decision',              'INTEGER NOT NULL'),
        ('review_time_posix',            '''INTEGER DEFAULT (CAST(STRFTIME('%s', 'NOW', 'UTC') AS INTEGER))'''),  # this should probably be UCT
        ('review_identity',              'TEXT'),
        ('review_tags',                  'TEXT'),
    ),
        superkeys=[('annot_1_rowid', 'annot_2_rowid', 'review_count')],
        docstr='''
        Used to store completed user review states of two matched annotations
        ''')


def update_1_0_1(db, ibs=None):
    db.modify_table(
        const.REVIEW_TABLE,
        add_columns=[
            ('review_user_confidence', 'INTEGER'),
        ],
        rename_columns=[
            ('review_identity', 'review_user_identity'),
        ]
    )

    pass

# ========================
# Valid Versions & Mapping
# ========================

# TODO: do we save a backup with the older version number in the file name?


base = const.BASE_DATABASE_VERSION
VALID_VERSIONS = ut.odict([
    #version:    (Pre-Update Function,  Update Function,    Post-Update Function)
    (base   ,    (None,                 None,               None                )),
    ('1.0.0',    (None,                 update_1_0_0,       None                )),
    ('1.0.1',    (None,                 update_1_0_1,       None                )),
])
"""
SeeAlso:
    When updating versions need to test and modify in
    IBEISController._init_sqldbcore
"""


LEGACY_UPDATE_FUNCTIONS = [
]


def autogen_staging_schema():
    """
    autogen_staging_schema

    CommandLine:
        python -m ibeis.control.STAGING_SCHEMA --test-autogen_staging_schema
        python -m ibeis.control.STAGING_SCHEMA --test-autogen_staging_schema --diff=1
        python -m ibeis.control.STAGING_SCHEMA --test-autogen_staging_schema -n=-1
        python -m ibeis.control.STAGING_SCHEMA --test-autogen_staging_schema -n=0
        python -m ibeis.control.STAGING_SCHEMA --test-autogen_staging_schema -n=1
        python -m ibeis.control.STAGING_SCHEMA --force-incremental-db-update
        python -m ibeis.control.STAGING_SCHEMA --test-autogen_staging_schema --write
        python -m ibeis.control.STAGING_SCHEMA --test-autogen_staging_schema --force-incremental-db-update --dump-autogen-schema
        python -m ibeis.control.STAGING_SCHEMA --test-autogen_staging_schema --force-incremental-db-update


    Example:
        >>> # ENABLE_DOCTEST
        >>> from ibeis.control.STAGING_SCHEMA import *  # NOQA
        >>> autogen_staging_schema()
    """
    from ibeis.control import STAGING_SCHEMA
    from ibeis.control import _sql_helpers
    n = ut.get_argval('-n', int, default=-1)
    schema_spec = STAGING_SCHEMA
    db = _sql_helpers.autogenerate_nth_schema_version(schema_spec, n=n)
    return db


if __name__ == '__main__':
    """
    python -m ibeis.algo.preproc.preproc_chip
    python -m ibeis.control.STAGING_SCHEMA --allexamples
    """
    import multiprocessing
    multiprocessing.freeze_support()
    import utool as ut
    ut.doctest_funcs()
