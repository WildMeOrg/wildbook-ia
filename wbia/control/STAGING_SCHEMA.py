# -*- coding: utf-8 -*-
"""
Module Licence and docstring

TODO: ideally the wbia.constants module would not be used here
and each function would use its own constant variables that are suffixed
with the last version number that they existed in

CommandLine:
    python -m wbia.control.STAGING_SCHEMA --test-autogen_staging_schema
"""
from __future__ import absolute_import, division, print_function
from wbia import constants as const
import utool as ut

(print, rrr, profile) = ut.inject2(__name__)

try:
    from wbia.control import STAGING_SCHEMA_CURRENT

    UPDATE_CURRENT = STAGING_SCHEMA_CURRENT.update_current
    VERSION_CURRENT = STAGING_SCHEMA_CURRENT.VERSION_CURRENT
except Exception:
    UPDATE_CURRENT = None
    VERSION_CURRENT = None
    print('[dbcache] NO STAGING_SCHEMA_CURRENT AUTO-GENERATED!')

profile = ut.profile


REVIEW_ROWID = 'review_rowid'
TEST_ROWID = 'test_rowid'


# =======================
# Schema Version 1.0.0
# =======================


@profile
def update_1_0_0(db, ibs=None):
    db.add_table(
        const.REVIEW_TABLE,
        (
            ('review_rowid', 'INTEGER PRIMARY KEY'),
            ('annot_1_rowid', 'INTEGER NOT NULL'),
            ('annot_2_rowid', 'INTEGER NOT NULL'),
            ('review_count', 'INTEGER NOT NULL'),
            ('review_decision', 'INTEGER NOT NULL'),
            (
                'review_time_posix',
                """INTEGER DEFAULT (CAST(STRFTIME('%s', 'NOW', 'UTC') AS INTEGER))""",
            ),  # this should probably be UCT
            ('review_identity', 'TEXT'),
            ('review_tags', 'TEXT'),
        ),
        superkeys=[('annot_1_rowid', 'annot_2_rowid', 'review_count')],
        docstr="""
        Used to store completed user review states of two matched annotations
        """,
    )


def update_1_0_1(db, ibs=None):
    db.modify_table(
        const.REVIEW_TABLE,
        add_columns=[('review_user_confidence', 'INTEGER')],
        rename_columns=[('review_identity', 'review_user_identity')],
    )


def update_1_0_2(db, ibs=None):
    db.modify_table(
        const.REVIEW_TABLE,
        (
            (1, 'review_uuid', 'UUID', None),
            (None, 'review_client_start_time_posix', 'INTEGER', None),
            (None, 'review_client_end_time_posix', 'INTEGER', None),
            (None, 'review_server_start_time_posix', 'INTEGER', None),
            ('review_time_posix', 'review_server_end_time_posix', 'INTEGER', None),
        ),
    )


def post_1_0_2(db, ibs=None):
    if ibs is not None:
        import uuid

        review_rowid_list = ibs._get_all_review_rowids()
        review_uuid_list = [uuid.uuid4() for _ in range(len(review_rowid_list))]
        ibs._set_review_uuids(review_rowid_list, review_uuid_list)
    db.modify_table(
        const.REVIEW_TABLE, [('review_uuid', '', 'UUID NOT NULL', None)],
    )


def update_1_0_3(db, ibs=None):
    db.modify_table(
        const.REVIEW_TABLE,
        (
            ('review_decision', 'review_evidence_decision', 'INTEGER', None),
            (None, 'review_meta_decision', 'INTEGER', None),
        ),
    )


def update_1_1_0(db, ibs=None):
    db.add_table(
        const.TEST_TABLE,
        (
            ('test_rowid', 'INTEGER PRIMARY KEY'),
            ('test_uuid', 'UUID'),
            ('test_user_identity', 'TEXT'),
            ('test_challenge_json', 'TEXT'),
            ('test_response_json', 'TEXT'),
            ('test_result', 'INTEGER'),
            (
                'test_time_posix',
                """INTEGER DEFAULT (CAST(STRFTIME('%s', 'NOW', 'UTC') AS INTEGER))""",
            ),  # this should probably be UCT
        ),
        superkeys=[('test_uuid',)],
        docstr="""
        Used to store tests given to the user, their responses, and their results
        """,
    )


def update_1_1_1(db, ibs=None):
    db.modify_table(const.REVIEW_TABLE, add_columns=[('review_metadata_json', 'TEXT')])


# ========================
# Valid Versions & Mapping
# ========================

# TODO: do we save a backup with the older version number in the file name?


base = const.BASE_DATABASE_VERSION
VALID_VERSIONS = ut.odict(
    [
        # version:    (Pre-Update Function,  Update Function,    Post-Update Function)
        (base, (None, None, None)),
        ('1.0.0', (None, update_1_0_0, None)),
        ('1.0.1', (None, update_1_0_1, None)),
        ('1.0.2', (None, update_1_0_2, post_1_0_2)),
        ('1.0.3', (None, update_1_0_3, None)),
        ('1.1.0', (None, update_1_1_0, None)),
        ('1.1.1', (None, update_1_1_1, None)),
    ]
)
"""
SeeAlso:
    When updating versions need to test and modify in
    IBEISController._init_sqldbcore
"""


LEGACY_UPDATE_FUNCTIONS = []


def autogen_staging_schema():
    """
    autogen_staging_schema

    CommandLine:
        python -m wbia.control.STAGING_SCHEMA --test-autogen_staging_schema
        python -m wbia.control.STAGING_SCHEMA --test-autogen_staging_schema --diff=1
        python -m wbia.control.STAGING_SCHEMA --test-autogen_staging_schema -n=-1
        python -m wbia.control.STAGING_SCHEMA --test-autogen_staging_schema -n=0
        python -m wbia.control.STAGING_SCHEMA --test-autogen_staging_schema -n=1
        python -m wbia.control.STAGING_SCHEMA --force-incremental-db-update
        python -m wbia.control.STAGING_SCHEMA --test-autogen_staging_schema --write
        python -m wbia.control.STAGING_SCHEMA --test-autogen_staging_schema --force-incremental-db-update --dump-autogen-schema
        python -m wbia.control.STAGING_SCHEMA --test-autogen_staging_schema --force-incremental-db-update


    Example:
        >>> # ENABLE_DOCTEST
        >>> from wbia.control.STAGING_SCHEMA import *  # NOQA
        >>> autogen_staging_schema()
    """
    from wbia.control import STAGING_SCHEMA
    from wbia.control import _sql_helpers

    n = ut.get_argval('-n', int, default=-1)
    schema_spec = STAGING_SCHEMA
    db = _sql_helpers.autogenerate_nth_schema_version(schema_spec, n=n)
    return db


if __name__ == '__main__':
    """
    python -m wbia.algo.preproc.preproc_chip
    python -m wbia.control.STAGING_SCHEMA --allexamples
    """
    import multiprocessing

    multiprocessing.freeze_support()
    import utool as ut

    ut.doctest_funcs()
