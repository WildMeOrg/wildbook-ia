"""
Module Licence and docstring
"""
from __future__ import absolute_import, division, print_function
from ibeis import constants
try:
    from ibeis.web import DBWEB_SCHEMA_CURRENT
    UPDATE_CURRENT  = DBWEB_SCHEMA_CURRENT.update_current
    VERSION_CURRENT = DBWEB_SCHEMA_CURRENT.VERSION_CURRENT
except:
    UPDATE_CURRENT  = None
    VERSION_CURRENT = None
    print("[dbcache] NO DBWEB_SCHEMA_CURRENT AUTO-GENERATED!")


VIEWPOINT_TABLE = 'viewpoints'
REVIEW_TABLE = 'reviews'


# =======================
# Schema Version 1.0.0
# =======================


def update_1_0_0(db, ibs=None):
    db.add_table(VIEWPOINT_TABLE, (
        ('viewpoint_rowid',              'INTEGER PRIMARY KEY'),
        ('annot_rowid',                  'INTEGER'),
        ('viewpoint_value_1',            'INTEGER DEFAULT -1'),
        ('viewpoint_value_2',            'INTEGER DEFAULT -1'),
        ('viewpoint_value_avg',          'INTEGER DEFAULT -1'),
    ),
        superkey_colnames=['annot_rowid'],
        docstr='''
        SQLite table to store the web state for viewpoint turking''')

    db.add_table(REVIEW_TABLE, (
        ('review_rowid',                 'INTEGER PRIMARY KEY'),
        ('image_rowid',                  'INTEGER'),
        ('review_count',                 'INTEGER DEFAULT 0'),
    ),
        superkey_colnames=['image_rowid'],
        docstr='''
        SQLite table to store the web state for detection review''')


# ========================
# Valid Versions & Mapping
# ========================


base = constants.BASE_DATABASE_VERSION
VALID_VERSIONS = {
    #version:   (Pre-Update Function,  Update Function,    Post-Update Function)
    base   :    (None,                 None,               None                ),
    '1.0.0':    (None,                 update_1_0_0,       None                ),
}
